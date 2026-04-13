import os
import streamlit as st
import tempfile
import json
from PIL import Image
import base64
import re
import pandas as pd
from sqlalchemy import create_engine, text

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_community.embeddings import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver # --- NEW: Persistent Memory ---
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage

# --- 1. UI Setup ---
st.set_page_config(page_title="Visual Data Worker Pro", page_icon="🛡️", layout="wide")
st.title("🛡️ Visual Data Worker: Enterprise Edition")
st.markdown("Security-First AI for SQL, RAG, and Automated Visualization.")

# --- 1.5 Enterprise UI Styling ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] { background-color: #0E1117; border-right: 1px solid #333; }
    [data-testid="stChatInput"] { border: 1px solid #4CAF50 !important; border-radius: 10px !important; }
    [data-testid="stChatMessage"]:nth-child(odd) { background-color: rgba(76, 175, 80, 0.1); border-left: 3px solid #4CAF50; }
    .streamlit-expanderHeader { font-weight: bold; color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- 2. Credentials ---
db_host = st.secrets["DATABRICKS_HOST"]
db_path = st.secrets["DATABRICKS_HTTP_PATH"]
db_token = st.secrets["DATABRICKS_TOKEN"]
databricks_uri = f"databricks://token:{db_token}@{db_host}:443?http_path={db_path}&catalog=data_agent_app&schema=data_agent"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- 3. Initialize Agent with HITL & Memory Saver ---
@st.cache_resource
def get_visual_agent():
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    db = SQLDatabase.from_uri(databricks_uri, sample_rows_in_table_info=0)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        safety_settings={cat: HarmBlockThreshold.BLOCK_NONE for cat in [
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_HARASSMENT, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT]}
    )
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    chart_holder = {}
    python_tool = PythonAstREPLTool(locals={"chart_holder": chart_holder})
    all_tools = toolkit.get_tools() + [python_tool]
    
    if "vector_store" in st.session_state and st.session_state["vector_store"] is not None:
        retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 4})
        all_tools.append(create_retriever_tool(retriever, "search_company_reports", "Search qualitative PDF info."))
    
    # --- UPDATED PREFIX: HITL & Data Audit Rules ---
    custom_prefix = """You are a Security-Focused C-Suite Data Analyst.
    1. SECURITY GATE: If you intend to use SQL 'INSERT', 'UPDATE', 'DELETE', or 'DROP', you MUST explicitly state "SECURITY_CONFIRMATION_REQUIRED" and show the query. Do NOT run it yet.
    2. DATA AUDIT: Before drawing any chart, you MUST audit the data. If values are NULL, None, or 0.0, you must handle them (filter them out or fill them) in Python before plotting.
    3. VISUALIZATION: Use plotly.express. Save to chart_holder['current_fig'].
    4. VOICE: Always explain your steps in plain English. Never return a blank response."""

    llm_with_tools = llm.bind_tools(all_tools)
    
    def call_model(state: AgentState):
        messages = [SystemMessage(content=custom_prefix)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}
        
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(all_tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", lambda state: "tools" if state["messages"][-1].tool_calls else END)
    workflow.add_edge("tools", "agent")
    
    # --- NEW: MemorySaver for Persistent Session Threads ---
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory), chart_holder

# --- Helper Function for DB Inserts ---
def databricks_insert(table, conn, keys, data_iter):
    values = []
    for row in data_iter: 
        formatted_row = []
        for val in row:
            if pd.isna(val):
                formatted_row.append("NULL")
            else:
                # We do the replacement outside the f-string to avoid backslash errors
                clean_val = str(val).replace("'", "''")
                formatted_row.append(f"'{clean_val}'")
        values.append(f"({', '.join(formatted_row)})")
    
    # Construct and execute the SQL
    sql_query = f"INSERT INTO {table.name} ({', '.join(keys)}) VALUES {', '.join(values)}"
    conn.execute(text(sql_query))

# --- 4. Sidebar: Omni-Channel Ingestion ---
with st.sidebar:
    st.header("📂 Data Ingestion")
    if "uploader_key" not in st.session_state: st.session_state["uploader_key"] = 0
    
    upload_type = st.selectbox("Select Upload Type:", ["CSV/Excel", "PDF Report", "Image/Receipt"])
    st.divider()

    if upload_type == "CSV/Excel":
        uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx"], key=f"sql_{st.session_state['uploader_key']}")
        if uploaded_file:
            engine = create_engine(databricks_uri)
            with st.spinner("Uploading..."):
                df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', uploaded_file.name.rsplit('.', 1)[0].lower())
                df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower() for col in df.columns]
                df.to_sql(clean_name, con=engine, if_exists="replace", index=False, method=databricks_insert)
            st.session_state["last_sql"] = clean_name
            st.session_state["uploader_key"] += 1
            get_visual_agent.clear(); st.rerun()

    elif upload_type == "PDF Report":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key=f"pdf_{st.session_state['uploader_key']}")
        if uploaded_file:
            with st.spinner("Vectorizing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    loader = PyPDFLoader(tmp.name)
                    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
                    st.session_state["vector_store"] = FAISS.from_documents(splits, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
            st.session_state["last_pdf"] = uploaded_file.name
            st.session_state["uploader_key"] += 1
            get_visual_agent.clear(); st.rerun()

    elif upload_type == "Image / Receipt (PNG/JPG)":
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg"], key=f"img_{st.session_state['uploader_key']}")
        if uploaded_file:
            with st.spinner("Vision Processing..."):
                img_b64 = base64.b64encode(uploaded_file.getvalue()).decode()
                v_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                res = v_llm.invoke([{"role": "user", "content": [{"type": "text", "text": "Extract table as JSON array. Keys: clean_lowercase."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}])
                data = json.loads(res.content.strip('`json\n '))
                df = pd.DataFrame(data)
                tbl_name = re.sub(r'[^a-zA-Z0-9_]', '_', uploaded_file.name.lower()) + "_img"
                df.to_sql(tbl_name, con=create_engine(databricks_uri), if_exists="replace", index=False, method=databricks_insert)
            st.session_state["last_img"] = tbl_name
            st.session_state["uploader_key"] += 1
            get_visual_agent.clear(); st.rerun()

    st.divider()
    st.markdown("##### 🧠 System Memory")
    if "last_sql" in st.session_state: st.success(f"📊 DB: {st.session_state['last_sql']}")
    if "last_pdf" in st.session_state: st.success(f"📄 RAG: {st.session_state['last_pdf']}")
    if "last_img" in st.session_state: st.success(f"👁️ Vision: {st.session_state['last_img']}")

# --- 5. Instantiate Agent ---
agent, chart_holder = get_visual_agent()
config = {"configurable": {"thread_id": "user_session_1"}} # --- Persistent Thread ID ---

# --- 6. Chat History ---
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("chart"): st.plotly_chart(msg["chart"], use_container_width=True)
        st.markdown(msg["content"])

# --- 7. Execution Loop with Persistent Pending Actions ---

# A. ALWAYS CHECK FOR PENDING ACTIONS FIRST (Outside the chat input)
if "pending_action" in st.session_state and st.session_state["pending_action"]:
    action = st.session_state["pending_action"]
    
    with st.chat_message("assistant"):
        st.warning("🚨 **Sensitive Action Detected!**")
        st.markdown(f"**Action Requested:**\n{action['display_msg']}")
        
        col1, col2 = st.columns(2)
        
        if col1.button("✅ Approve & Run"):
            try:
                engine = create_engine(databricks_uri)
                # Use a context manager to ensure the connection is closed and committed
                with engine.begin() as conn: 
                    conn.execute(text(action['sql']))
                
                # --- THE FIX: CLEAR THE CACHE ---
                get_visual_agent.clear() 
                
                success_msg = "✅ **Confirmation received. The table has been permanently deleted, and the agent's memory has been refreshed.**"
                st.balloons()
                
                # Save to history so the AGENT sees this message in the next turn
                st.session_state.messages.append({"role": "assistant", "content": success_msg})
                
                st.session_state["pending_action"] = None
                st.rerun() 
            except Exception as e:
                st.error(f"Execution Error: {e}")
                st.session_state["pending_action"] = None

        if col2.button("❌ Reject"):
            st.info("Action cancelled.")
            st.session_state["pending_action"] = None
            st.rerun()

# B. THE MAIN CHAT INPUT
if prompt := st.chat_input("E.g., Draw a chart of IMDb ratings, filter out nulls."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing & Auditing..."):
            chart_holder.clear()
            result = agent.invoke({"messages": [("user", prompt)]}, config=config)
            
            # 1. Clean Text Extraction
            raw_content = result["messages"][-1].content
            final_text = ""
            if isinstance(raw_content, list):
                for item in raw_content:
                    if isinstance(item, dict): final_text += item.get('text', '')
                    else: final_text += str(item)
            else:
                final_text = str(raw_content)

            # 2. Check for Security Trigger
            if "SECURITY_CONFIRMATION_REQUIRED" in final_text:
                sql_match = re.search(r"```sql\n(.*?)\n```", final_text, re.DOTALL)
                clean_msg = final_text.replace("SECURITY_CONFIRMATION_REQUIRED", "").strip()
                
                if sql_match:
                    # Capture the action for the next rerun
                    st.session_state["pending_action"] = {
                        "sql": sql_match.group(1),
                        "display_msg": clean_msg
                    }
                    st.rerun() # Trigger the rerun immediately to show the buttons
                else:
                    st.error("Sensitive action requested but no SQL found.")
            
            # 3. Normal Response Rendering
            else:
                gen_chart = chart_holder.get('current_fig')
                if gen_chart: 
                    st.plotly_chart(gen_chart, use_container_width=True)
                st.markdown(final_text)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_text, 
                    "chart": gen_chart
                })

            # 4. Audit Trail
            with st.expander("🔍 Audit Trail & Thought Process"):
                for m in result["messages"]:
                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        for tc in m.tool_calls: 
                            st.code(f"Tool: {tc['name']}\nArgs: {tc['args']}")