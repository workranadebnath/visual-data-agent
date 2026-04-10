import os
import streamlit as st
import tempfile

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, HarmCategory, HarmBlockThreshold
from langchain_experimental.tools import PythonAstREPLTool

# --- NEW: RAG Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage

import pandas as pd
from sqlalchemy import create_engine, text
import re

# --- 1. UI Setup ---
st.set_page_config(page_title="Visual Data Worker", page_icon="📈", layout="wide")
st.title("📈 Visual Data Worker By Rana Debnath")
st.markdown("I can query databases, draw charts, and **read qualitative PDF reports.**")

# --- 2. Credentials ---
db_host = st.secrets["DATABRICKS_HOST"]
db_path = st.secrets["DATABRICKS_HTTP_PATH"]
db_token = st.secrets["DATABRICKS_TOKEN"]

databricks_uri = f"databricks://token:{db_token}@{db_host}:443?http_path={db_path}&catalog=data_agent_app&schema=data_agent"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- 3. Initialize Explicit LangGraph with RAG ---
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
    
    # --- NEW: Inject the Retriever Tool if a PDF was uploaded ---
    if "vector_store" in st.session_state and st.session_state["vector_store"] is not None:
        retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 4})
        retriever_tool = create_retriever_tool(
            retriever,
            "search_company_reports",
            "Searches and returns qualitative information, text, and context from the uploaded PDF reports. Use this when the user asks about strategy, reasons, notes, or text-based documents."
        )
        all_tools.append(retriever_tool)
    
    # --- UPDATED: The Omni-Agent Prompt ---
    custom_prefix = """You are an elite C-Suite Data Analyst. You have 3 main capabilities:
    1. SQL DATABASE: Use SQL tools to extract quantitative numbers from Databricks.
    2. PDF DOCUMENTS: Use the 'search_company_reports' tool to find qualitative context, strategies, or explanations from uploaded PDFs.
    3. DATA VISUALIZATION: Use the 'python_repl_ast' tool to draw charts.
    
    THE PLOTLY RULE: You must use plotly.express to draw charts. NEVER use matplotlib.
    THE STREAMLIT RULE: To show the chart, you MUST save the figure to Streamlit's session state under the exact key 'current_fig'.
    THE HARDCODE RULE: The Python environment DOES NOT have access to the SQL tool's output variables. You MUST hardcode the data arrays explicitly.
    THE SELF-HEALING RULE: If your python tool returns an error, read the error, fix your code, and run it again.
    
    Example Python code:
    import plotly.express as px
    import pandas as pd
    import streamlit as st
    
    shows = ['A', 'B', 'C']
    hours = [10, 20, 30]
    df = pd.DataFrame({{'Show': shows, 'Hours': hours}})
    fig = px.bar(df, x='Hours', y='Show')
    st.session_state['current_fig'] = fig
    
    Always combine context from the PDFs with hard data from the database if the user asks for both!"""

    llm_with_tools = llm.bind_tools(all_tools)
    
    def call_model(state: AgentState):
        messages = [SystemMessage(content=custom_prefix)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    tool_node = ToolNode(all_tools)
    
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
        
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# --- 4. Sidebar: Omni-Channel Ingestion ---
with st.sidebar:
    st.header("📂 Enterprise Data Ingestion")
    
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
        
    # --- SQL UPLOADER ---
    st.subheader("1. Quantitative Data (DB)")
    uploaded_sql_file = st.file_uploader(
        "Upload Datasets (CSV/Excel)", 
        type=["csv", "xlsx"], 
        key=f"sql_uploader_{st.session_state['uploader_key']}"
    )
    
    if uploaded_sql_file is not None:
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

        with st.spinner(f"Uploading '{uploaded_sql_file.name}' to Databricks..."):
            if uploaded_sql_file.name.endswith('.xlsx'):
                excel_data = pd.read_excel(uploaded_sql_file, sheet_name=None)
                for sheet_name, df in excel_data.items():
                    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name.lower())
                    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower() for col in df.columns]
                    df.to_sql(clean_name, con=engine, if_exists="replace", index=False, chunksize=2000, method=databricks_insert)
            elif uploaded_sql_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_sql_file)
                raw_name = uploaded_sql_file.name.rsplit('.', 1)[0]
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name.lower())
                df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower() for col in df.columns]
                df.to_sql(clean_name, con=engine, if_exists="replace", index=False, chunksize=2000, method=databricks_insert)

        st.session_state["last_uploaded_sql"] = uploaded_sql_file.name
        get_visual_agent.clear() 
        st.session_state["uploader_key"] += 1
        st.rerun()

    if "last_uploaded_sql" in st.session_state:
        st.success(f"✅ DB: `{st.session_state['last_uploaded_sql']}` loaded!")

    st.divider()

    # --- NEW: PDF VECTOR UPLOADER ---
    st.subheader("2. Qualitative Data (RAG)")
    uploaded_pdf_file = st.file_uploader(
        "Upload Reports (PDF)", 
        type=["pdf"], 
        key=f"pdf_uploader_{st.session_state['uploader_key']}"
    )

    if uploaded_pdf_file is not None:
        with st.spinner(f"Reading and vectorizing '{uploaded_pdf_file.name}'..."):
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
            
            # Save PDF temporarily to parse it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract and chunk text
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(pages)
            
            # --- NEW: Safety net for empty or scanned PDFs ---
            if not splits:
                st.error("⚠️ Could not extract any text from this PDF. Please ensure it is a text-based PDF (where you can highlight words) and not a scanned image.")
                os.remove(tmp_file_path)
                st.stop()
            
            # Build FAISS Vector Database using Google Embeddings
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=st.secrets["GOOGLE_API_KEY"]
                )
                vectorstore = FAISS.from_documents(splits, embeddings)
            except Exception as e:
                st.error(f"🛑 Google API Rejected the PDF. Exact Reason: {e}")
                os.remove(tmp_file_path)
                st.stop()
            
            # Save to session memory and clear temp file
            st.session_state["vector_store"] = vectorstore
            os.remove(tmp_file_path)

        st.session_state["last_uploaded_pdf"] = uploaded_pdf_file.name
        get_visual_agent.clear() # Clear agent cache so it loads the new tool
        st.session_state["uploader_key"] += 1
        st.rerun()

    if "last_uploaded_pdf" in st.session_state:
        st.success(f"✅ RAG: `{st.session_state['last_uploaded_pdf']}` vectorized!")

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
if prompt := st.chat_input("E.g., Based on the PDF, why did costs rise? Draw a chart of actual vs budget."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.container()
        with st.spinner("Synthesizing data, writing code, and self-correcting..."):
            try:
                if 'current_fig' in st.session_state:
                    del st.session_state['current_fig']
                
                lg_messages = []
                if len(st.session_state.messages) > 1:
                    for msg in st.session_state.messages[-5:-1]:
                        lg_messages.append((msg["role"], msg["content"]))
                
                lg_messages.append(("user", prompt))

                response = agent.invoke({"messages": lg_messages})
                
                raw_content = response["messages"][-1].content
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