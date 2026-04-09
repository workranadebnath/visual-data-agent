import os
import streamlit as st
import matplotlib.pyplot as plt
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonAstREPLTool

# --- 1. UI Setup ---
st.set_page_config(page_title="Visual Data Worker", page_icon="📈")
st.title("📈 Visual Data Worker")
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
if prompt := st.chat_input("E.g., Draw a bar chart of total revenue by product category"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We use a container to catch both text and charts
        container = st.container()
        with st.spinner("Generating visualization..."):
            try:
                # We use a trick here: we capture the current matplotlib figure
                response = agent.invoke({"input": prompt})
                
                # Check if a plot was created in the global matplotlib state
                fig = plt.gcf()
                
                # If the figure has axes (meaning something was drawn)
                if fig.get_axes():
                    st.pyplot(fig)
                    plt.clf() # Clear for the next run
                
                final_text = response.get('output', str(response))
                st.markdown(final_text)
                
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                
            except Exception as e:
                st.error(f"Error: {e}")
