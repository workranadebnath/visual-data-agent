import os
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_experimental.tools import PythonAstREPLTool

# --- NEW: We import the raw LangGraph components instead of the broken helper ---
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage

import pandas as pd
from sqlalchemy import create_engine, text
import re

# --- 1. UI Setup ---
st.set_page_config(page_title="Visual Data Worker", page_icon="📈")
st.title("📈 Visual Data Worker By Rana Debnath")
st.markdown("I can now query your data, draw charts, and **self-heal my own code.**")

# --- 2. Credentials ---
db_host = st.secrets["DATABRICKS_HOST"]
db_path = st.secrets["DATABRICKS_HTTP_PATH"]
db_token = st.secrets["DATABRICKS_TOKEN"]

databricks_uri = f"databricks://token:{db_token}@{db_host}:443?http_path={db_path}&catalog=data_agent_app&schema=data_agent"

# --- NEW: Define the Graph's Memory State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- 3. Initialize Explicit LangGraph ---
@st.cache_resource
def get_visual_agent():
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    db = SQLDatabase.from_uri(
        databricks_uri,
        sample_rows_in_table_info=0 
    )
    
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
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = toolkit.get_tools()
    python_tool = PythonAstREPLTool()
    
    all_tools = sql_tools + [python_tool]
    
    custom_prefix = """You are an expert Visual Data Analyst.
    1. For data questions, use the SQL tools to find the answer.
    2. If the user asks for a chart, you must use the python_repl_ast tool.
    3. THE PLOTLY RULE: You must use plotly.express to draw charts. NEVER use matplotlib.
    4. THE STREAMLIT RULE: To show the chart to the user, you MUST save the figure to Streamlit's session state under the exact key 'current_fig'.
    5. THE HARDCODE RULE: The Python environment DOES NOT have access to the SQL tool's output variables. You MUST hardcode the data arrays explicitly.
    6. THE SELF-HEALING RULE: If your python_repl_ast tool returns an error, read the error, fix your code, and run the tool again until it succeeds!
    
    Example of correct Python code:
    import plotly.express as px
    import pandas as pd
    import streamlit as st
    
    shows = ['Stranger Things', 'Bridgerton', 'Ozark']
    hours = [1580910000, 710050000, 281460000]
    df = pd.DataFrame({{'Show': shows, 'Hours': hours}})
    
    fig = px.bar(df, x='Hours', y='Show', orientation='h', title='Top Shows')
    st.session_state['current_fig'] = fig
    
    7. Always return a plain-English explanation of the final chart."""

    # --- NEW: Build the State Machine Explicitly ---
    
    # 1. Give the AI its tools
    llm_with_tools = llm.bind_tools(all_tools)
    
    # 2. Define the "Thinking" Node
    def call_model(state: AgentState):
        messages = [SystemMessage(content=custom_prefix)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    # 3. Define the "Acting" Node
    tool_node = ToolNode(all_tools)
    
    # 4. Define the routing logic (The Self-Healing Loop)
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        # If the AI decided to use a tool, route to the tool node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # Otherwise, the task is finished
        return END
        
    # 5. Wire the graph together
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# --- 4. Sidebar: Enterprise Data Ingestion ---
with st.sidebar:
    st.header("📂 Enterprise Data Ingestion")
    
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    
    uploaded_file = st.file_uploader(
        "Upload Data (CSV or Excel)", 
        type=["csv", "xlsx"], 
        key=f"uploader_{st.session_state['uploader_key']}"
    )
    
    if uploaded_file is not None:
        engine = create_engine(databricks_uri)
        
        def databricks_insert(table, conn, keys, data_iter):
            values = []
            for row in data_iter: 
                formatted_row = []
                for val in row:   
                    if pd.isna(val):
                        formatted_row.append("NULL")
                    else:
                        val_str = str(val).replace("'", "''")
                        formatted_row.append(f"'{val_str}'")
                values.append(f"({', '.join(formatted_row)})")
            
            sql = f"INSERT INTO {table.name} ({', '.join(keys)}) VALUES {', '.join(values)}"
            conn.execute(text(sql))

        with st.spinner(f"Uploading '{uploaded_file.name}' to Databricks..."):
            if uploaded_file.name.endswith('.xlsx'):
                excel_data = pd.read_excel(uploaded_file, sheet_name=None)
                for sheet_name, df in excel_data.items():
                    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name.lower())
                    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower() for col in df.columns]
                    df.to_sql(clean_name, con=engine, if_exists="replace", index=False, chunksize=2000, method=databricks_insert)
            
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                raw_name = uploaded_file.name.rsplit('.', 1)[0]
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name.lower())
                df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower() for col in df.columns]
                df.to_sql(clean_name, con=engine, if_exists="replace", index=False, chunksize=2000, method=databricks_insert)

        st.session_state["last_uploaded_file"] = uploaded_file.name
        get_visual_agent.clear() 
        st.session_state["uploader_key"] += 1
        st.rerun()

    if "last_uploaded_file" in st.session_state:
        st.success(f"✅ `{st.session_state['last_uploaded_file']}` successfully loaded to Databricks!")
        
# --- 5. Instantiate Agent ---
agent = get_visual_agent()

# --- 6. Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "chart" in message and message["chart"] is not None:
            st.plotly_chart(message["chart"], use_container_width=True)
        st.markdown(message["content"])

# --- 7. Input & Execution ---
if prompt := st.chat_input("E.g., Draw a bar chart of total revenue by product category"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.container()
        with st.spinner("Analyzing database, writing code, and self-correcting..."):
            try:
                if 'current_fig' in st.session_state:
                    del st.session_state['current_fig']
                
                lg_messages = []
                if len(st.session_state.messages) > 1:
                    for msg in st.session_state.messages[-5:-1]:
                        lg_messages.append((msg["role"], msg["content"]))
                
                lg_messages.append(("user", prompt))

                # --- Execute the Graph ---
                response = agent.invoke({"messages": lg_messages})
                
                # Extract the raw content from the last message
                raw_content = response["messages"][-1].content
                
                # --- NEW: Robust parser to clean up Gemini's raw dictionary blocks ---
                final_text = ""
                if isinstance(raw_content, list):
                    for item in raw_content:
                        if isinstance(item, dict):
                            final_text += item.get('text', '')
                        else:
                            final_text += str(item)
                else:
                    final_text = str(raw_content)
                
                generated_chart = st.session_state.get('current_fig', None)
                
                if generated_chart:
                    st.plotly_chart(generated_chart, use_container_width=True)
                
                st.markdown(final_text)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_text,
                    "chart": generated_chart
                })
                
            except Exception as e:
                st.error(f"Error: {e}")