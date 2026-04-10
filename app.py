import os
import streamlit as st
import matplotlib
matplotlib.use('Agg') # THIS IS THE FIX FOR CLOUD DEPLOYMENTS
import matplotlib.pyplot as plt
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_experimental.tools import PythonAstREPLTool

import pandas as pd
from sqlalchemy import create_engine, text # Add 'text' here
import re

# --- 1. UI Setup ---
st.set_page_config(page_title="Visual Data Worker", page_icon="📈")
st.title("📈 Visual Data Worker By Rana Debnath")
st.markdown("I can now query your data **and** draw charts.")

# --- 2. Sidebar: Enterprise Data Ingestion ---
# MOVED OUTSIDE THE CACHED FUNCTION TO FIX WARNING
db_host = st.secrets["DATABRICKS_HOST"]
db_path = st.secrets["DATABRICKS_HTTP_PATH"]
db_token = st.secrets["DATABRICKS_TOKEN"]

# Build the Databricks URI
databricks_uri = f"databricks://token:{db_token}@{db_host}:443?http_path={db_path}&catalog=data_agent_app&schema=data_agent"

with st.sidebar:
    st.header("📂 Enterprise Data Ingestion")
    
    # Allow both CSV and Excel uploads
    uploaded_file = st.file_uploader("Upload Data (CSV or Excel)", type=["csv", "xlsx"])
    
    # 🛑 SAFETY GUARD: Stop here if no file is uploaded yet
    if uploaded_file is not None:
        # Create a connection engine to your Databricks
        engine = create_engine(databricks_uri)
        
        # --- NEW: The Bulletproof Custom Insert Method ---
        def databricks_insert(table, conn, keys, data_iter):
            """Bypasses Databricks parameter bugs by forcing raw SQL strings"""
            for data in data_iter:
                values = []
                for row in data:
                    formatted_row = []
                    for val in row:
                        if pd.isna(val):
                            formatted_row.append("NULL")
                        else:
                            # Wrap everything in quotes and escape single quotes safely
                            val_str = str(val).replace("'", "''")
                            formatted_row.append(f"'{val_str}'")
                    values.append(f"({', '.join(formatted_row)})")
                
                # Build and execute the raw SQL string
                sql = f"INSERT INTO {table.name} ({', '.join(keys)}) VALUES {', '.join(values)}"
                conn.execute(text(sql))
        # ------------------------------------------------

        with st.spinner("Processing and uploading to Databricks..."):
            
            # --- EXCEL HANDLING ---
            if uploaded_file.name.endswith('.xlsx'):
                excel_data = pd.read_excel(uploaded_file, sheet_name=None)
                st.info(f"Found {len(excel_data)} sheets. Uploading...")
                
                for sheet_name, df in excel_data.items():
                    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name.lower())
                    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower() for col in df.columns]
                    
                    # Note the new method=databricks_insert argument!
                    df.to_sql(clean_name, con=engine, if_exists="replace", index=False, chunksize=100, method=databricks_insert)
                    st.success(f"✅ Sheet '{sheet_name}' saved as table `{clean_name}`")
            
            # --- CSV HANDLING ---
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                raw_name = uploaded_file.name.rsplit('.', 1)[0]
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name.lower())
                df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower() for col in df.columns]
                
                # Note the new method=databricks_insert argument!
                df.to_sql(clean_name, con=engine, if_exists="replace", index=False, chunksize=100, method=databricks_insert)
                st.success(f"✅ File saved as table `{clean_name}`")

        st.caption("Upload complete. You can now chat with this data.")

# --- 3. Initialize Agent ---
@st.cache_resource
def get_visual_agent():
    # Replace with your actual key
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    # Connect to Databricks
    db = SQLDatabase.from_uri(
        databricks_uri,
        sample_rows_in_table_info=0 # Set to 0 to bypass the Databricks LIMIT bug
    )
    
    # We use Gemini 2.5 Flash, but we turn down the default safety filters 
    # to prevent false positives from blocking our database queries.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    # The Python tool that allows the agent to draw
    python_tool = PythonAstREPLTool()
    
    custom_prefix = """You are an expert Visual Data Analyst.
    1. For data questions, write SQL to find the answer.
    2. If the user asks for a chart, you must use the python_repl_ast tool.
    3. THE PYTHON REPL RULE: The Python environment DOES NOT have access to the SQL tool's output variables. You MUST hardcode the data you found from the SQL query directly into your Python code.
    
    Example of correct Python code:
    import matplotlib.pyplot as plt
    import pandas as pd
    # You must type out the data arrays explicitly!
    shows = ['Stranger Things', 'Bridgerton', 'Ozark']
    hours = [1580910000, 710050000, 281460000]
    df = pd.DataFrame({{'Show': shows, 'Hours': hours}})
    fig, ax = plt.subplots()
    ax.barh(df['Show'], df['Hours'])
    
    4. NEVER call plt.show().
    5. Always return a plain-English explanation of the final chart."""

    return create_sql_agent(
        llm, 
        db=db, 
        agent_type="tool-calling", 
        verbose=True, 
        prefix=custom_prefix,
        extra_tools=[python_tool] # Add the drawing tool here
    )

agent = get_visual_agent()

# --- 4. Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Input & Execution ---
if prompt := st.chat_input("E.g., Draw a bar chart of total revenue by product category"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.container()
        with st.spinner("Analyzing database and conversation history..."):
            try:
                # --- MEMORY LOGIC ---
                chat_history = ""
                if len(st.session_state.messages) > 1:
                    chat_history = "Context from recent conversation:\n"
                    for msg in st.session_state.messages[-5:-1]: # Get previous messages excluding the current prompt
                        role = "User" if msg["role"] == "user" else "Data Worker"
                        chat_history += f"{role}: {msg['content']}\n"
                
                # Combine the memory with the user's new question
                contextual_prompt = f"{chat_history}\n\nNew Request: {prompt}"

                # Pass the combined prompt to the agent
                response = agent.invoke({"input": contextual_prompt})
                
                # Plotting logic
                fig = plt.gcf()
                if fig.get_axes():
                    st.pyplot(fig, use_container_width=True)
                    plt.clf() 
                
                # --- Bulletproof Output Parser V2 ---
                raw_output = response.get('output', '')
                final_text = ""

                # If it's a list, loop through and grab all the text pieces
                if isinstance(raw_output, list):
                    for item in raw_output:
                        if isinstance(item, dict):
                            final_text += item.get('text', '')
                        elif isinstance(item, str):
                            final_text += item
                else:
                    # If it's already just a normal string, use it directly
                    final_text = str(raw_output)
                    
                # Clean up any leftover weird formatting
                final_text = final_text.strip()
                
                st.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                
            except Exception as e:
                st.error(f"Error: {e}")