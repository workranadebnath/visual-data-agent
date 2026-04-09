import os
import streamlit as st
import matplotlib.pyplot as plt
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonAstREPLTool

# --- 1. UI Setup ---
st.set_page_config(page_title="Visual Data Worker", page_icon="📈")
st.title("📈 Visual Data Worker By Rana Debnath")
st.markdown("I can now query your data **and** draw charts.")

# --- 2. Initialize Agent ---
@st.cache_resource
def get_visual_agent():
    # Replace with your actual key
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    db = SQLDatabase.from_uri(
        "sqlite:///enterprise_data.db", 
        include_tables=['employees', 'customers', 'products', 'sales']
    )
    
    # We use Gemini 1.5 Pro for visualization as it is better at writing plotting code
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # NEW: The Python tool that allows the agent to draw
    python_tool = PythonAstREPLTool()
    
    custom_prefix = """You are a Visual Data Analyst.
    1. For data questions, write SQL.
    2. If the user asks for a chart, graph, or visualization, use the python_repl_ast tool.
    3. When plotting:
       - Use matplotlib.pyplot.
       - NEVER call plt.show(). Instead, the chart will be captured by the environment.
       - Make sure to label your axes and give the chart a title.
    4. If you use data from the database in a chart, query the data first, then pass it to the python tool."""

    return create_sql_agent(
        llm, 
        db=db, 
        agent_type="tool-calling", 
        verbose=True, 
        prefix=custom_prefix,
        extra_tools=[python_tool] # Add the drawing tool here
    )

agent = get_visual_agent()

# --- 3. Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Input & Execution ---
# --- 4. Input & Execution (Upgraded with Memory) ---
if prompt := st.chat_input("E.g., Draw a bar chart of total revenue by product category"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.container()
        with st.spinner("Analyzing database and conversation history..."):
            try:
                # --- NEW MEMORY LOGIC ---
                # We grab the last 4 messages from Streamlit's history so the agent remembers context
                # We limit it to 4 to save API tokens and keep the agent focused
                chat_history = ""
                if len(st.session_state.messages) > 1:
                    chat_history = "Context from recent conversation:\n"
                    for msg in st.session_state.messages[-5:-1]: # Get previous messages excluding the current prompt
                        role = "User" if msg["role"] == "user" else "Data Worker"
                        chat_history += f"{role}: {msg['content']}\n"
                
                # Combine the memory with the user's new question
                contextual_prompt = f"{chat_history}\n\nNew Request: {prompt}"
                # -------------------------

                # We pass the combined prompt to the agent
                response = agent.invoke({"input": contextual_prompt})
                
                # Plotting logic (same as before)
                fig = plt.gcf()
                if fig.get_axes():
                    st.pyplot(fig)
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
                # ------------------------------------
                
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                
            except Exception as e:
                st.error(f"Error: {e}")