import os
import streamlit as st
import tempfile
import json
from PIL import Image
import base64
import re
import time
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, HarmCategory, HarmBlockThreshold
from langchain_experimental.tools import PythonAstREPLTool

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_community.embeddings import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage

# --- 1. UI Setup ---
st.set_page_config(page_title="Visual Data Worker", page_icon="📈", layout="wide")
st.title("📈 Visual Data Worker By Rana Debnath")
st.markdown("I can query databases, draw charts, and **read qualitative PDF reports.**")

# --- 1.5 Enterprise UI Styling ---
st.markdown("""
<style>
    /* Hide Streamlit Default Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Make the Sidebar look distinct */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #333;
    }
    
    /* Style the Chat Input Box */
    [data-testid="stChatInput"] {
        border: 1px solid #4CAF50 !important;
        border-radius: 10px !important;
    }
    
    /* Give user messages a subtle highlight */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 3px solid #4CAF50;
    }
    
    /* Clean up the expander boxes */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Credentials ---
db_host = st.secrets["DATABRICKS_HOST"]
db_path = st.secrets["DATABRICKS_HTTP_PATH"]
db_token = st.secrets["DATABRICKS_TOKEN"]

databricks_uri = f"databricks://token:{db_token}@{db_host}:443?http_path={db_path}&catalog=data_agent_app&schema=data_agent"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- 3. Initialize Explicit LangGraph with Shared Memory ---
@st.cache_resource
def get_visual_agent():
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    # Track active tables to prevent schema bloat
    active_tables = []
    if "last_uploaded_sql" in st.session_state: active_tables.append(st.session_state["last_uploaded_sql"])
    if "last_uploaded_img" in st.session_state: active_tables.append(st.session_state["last_uploaded_img"])

    engine_kwargs = {"poolclass": NullPool}

    if active_tables:
        db = SQLDatabase.from_uri(
            databricks_uri,
            include_tables=active_tables,
            sample_rows_in_table_info=0,
            engine_args=engine_kwargs
        )
    else:
        db = SQLDatabase.from_uri(
            databricks_uri,
            sample_rows_in_table_info=0,
            engine_args=engine_kwargs
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
    
    chart_holder = {}
    python_tool = PythonAstREPLTool(
        locals={"chart_holder": chart_holder},
        description="CRITICAL: You MUST use this tool to execute Python code whenever a user asks for a chart, plot, or graph. Do NOT skip this step. Input must be valid Python code using pandas and plotly.express."
    )
    
    all_tools = sql_tools + [python_tool]
    
    if "vector_store" in st.session_state and st.session_state["vector_store"] is not None:
        retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 4})
        retriever_tool = create_retriever_tool(
            retriever,
            "search_company_reports",
            "Searches and returns qualitative information, text, and context from the uploaded PDF reports. Use this when the user asks about strategy, reasons, notes, or text-based documents."
        )
        all_tools.append(retriever_tool)
    
    tick3 = '```'
    custom_prefix = f"""You are an elite C-Suite Data Analyst. You have 3 main capabilities:
    1. SQL DATABASE: Use SQL tools to extract quantitative numbers from Databricks. You MUST use Databricks Spark SQL syntax.
    2. PDF DOCUMENTS: Use the 'search_company_reports' tool to find qualitative context, strategies, or explanations from uploaded PDFs.
    3. DATA VISUALIZATION: Use the 'python_repl_ast' tool to draw charts.
    
    SECURITY GATE (CRITICAL): You are STRICTLY FORBIDDEN from running 'INSERT', 'UPDATE', 'DELETE', or 'DROP' directly via tools. If the user asks you to modify data, reply ONLY with this exact format:
    SECURITY_CONFIRMATION_REQUIRED
    {tick3}sql
    [query]
    {tick3}
    
    THE PLOTLY RULE: You must use plotly.express to draw charts. NEVER use matplotlib.
    THE MEMORY RULE: To show the chart to the user, you MUST save the figure to the globally injected `chart_holder` dictionary under the exact key 'current_fig'. Do NOT use st.session_state!
    THE HARDCODE RULE: The Python environment DOES NOT have access to the SQL tool's output variables. You MUST hardcode the data arrays explicitly.
    THE SELF-HEALING RULE: If your python tool returns an error, read the error, fix your code, and run it again.
    THE STRICT VISUALIZATION RULE: If the user asks for ANY kind of chart, graph, or plot, you MUST use the python_repl_ast tool to generate it. You are strictly forbidden from saying "Here is the chart" unless you have explicitly executed the Python code to create it!    
    THE VOICE RULE: You must ALWAYS provide a plain-English explanation of your findings. Never return a blank response!
    
    Example Python code:
    {tick3}python
    import plotly.express as px
    import pandas as pd
    
    shows = ['A', 'B', 'C']
    hours = [10, 20, 30]
    df = pd.DataFrame({{'Show': shows, 'Hours': hours}})
    fig = px.bar(df, x='Hours', y='Show')
    chart_holder['current_fig'] = fig
    {tick3}
    
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
    
    return workflow.compile(), chart_holder

# --- Helper Function for DB Inserts ---
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
    
    # Bulletproof syntax for Databricks dynamic naming
    escaped_keys = [f"`{k}`" for k in keys]
    sql_query = f"INSERT INTO `{table.name}` ({', '.join(escaped_keys)}) VALUES {', '.join(values)}"
    conn.execute(text(sql_query))

# --- 4. Sidebar: Omni-Channel Ingestion ---
with st.sidebar:
    st.header("📂 Enterprise Data Ingestion")
    
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
        
    upload_type = st.selectbox(
        "Select Data Type to Upload:",
        ["Structured Data (CSV/Excel)", "Qualitative Report (PDF)", "Image / Receipt (PNG/JPG)"]
    )
    
    st.divider()

    # --- Option 1: SQL/CSV/Excel ---
    if upload_type == "Structured Data (CSV/Excel)":
        st.subheader("Quantitative Data (DB)")
        uploaded_sql_file = st.file_uploader(
            "Upload Datasets (CSV/Excel)", 
            type=["csv", "xlsx"], 
            key=f"sql_uploader_{st.session_state['uploader_key']}"
        )
        
        if uploaded_sql_file is not None:
            engine = create_engine(databricks_uri, poolclass=NullPool)
            with st.spinner(f"Uploading '{uploaded_sql_file.name}' to Databricks..."):
                if uploaded_sql_file.name.endswith('.xlsx'):
                    excel_data = pd.read_excel(uploaded_sql_file, sheet_name=None)
                    for sheet_name, df in excel_data.items():
                        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                        file_brief = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name.lower())[:15]
                        clean_name = f"table_{timestamp}_{file_brief}"
                        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower() for col in df.columns]
                        df.to_sql(clean_name, con=engine, if_exists="replace", index=False, chunksize=2000, method=databricks_insert)
                        st.session_state["last_uploaded_sql"] = clean_name
                elif uploaded_sql_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_sql_file)
                    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    file_brief = re.sub(r'[^a-zA-Z0-9_]', '_', uploaded_sql_file.name.rsplit('.', 1)[0].lower())[:15]
                    clean_name = f"table_{timestamp}_{file_brief}"
                    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower() for col in df.columns]
                    df.to_sql(clean_name, con=engine, if_exists="replace", index=False, chunksize=2000, method=databricks_insert)
                    st.session_state["last_uploaded_sql"] = clean_name

            get_visual_agent.clear() 
            st.session_state["uploader_key"] += 1
            st.rerun()

    # --- Option 2: PDF RAG ---
    elif upload_type == "Qualitative Report (PDF)":
        st.subheader("Qualitative Data (RAG)")
        uploaded_pdf_file = st.file_uploader(
            "Upload Reports (PDF)", 
            type=["pdf"], 
            key=f"pdf_uploader_{st.session_state['uploader_key']}"
        )

        if uploaded_pdf_file is not None:
            with st.spinner(f"Reading and vectorizing '{uploaded_pdf_file.name}'..."):
                os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(pages)
                
                if not splits:
                    st.error("⚠️ Could not extract any text from this PDF. Please ensure it is a text-based PDF (where you can highlight words) and not a scanned image.")
                    os.remove(tmp_file_path)
                    st.stop()
                
                try:
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(splits, embeddings)
                except Exception as e:
                    st.error(f"🛑 Vectorization Failed: {e}")
                    os.remove(tmp_file_path)
                    st.stop()
                
                st.session_state["vector_store"] = vectorstore
                os.remove(tmp_file_path)

            st.session_state["last_uploaded_pdf"] = uploaded_pdf_file.name
            get_visual_agent.clear() 
            st.session_state["uploader_key"] += 1
            st.rerun()

    # --- Option 3: Vision ---
    elif upload_type == "Image / Receipt (PNG/JPG)":
        st.subheader("Image Extraction (Vision to DB)")
        uploaded_img_file = st.file_uploader(
            "Upload Receipts/Charts (PNG/JPG)", 
            type=["png", "jpg", "jpeg"], 
            key=f"img_uploader_{st.session_state['uploader_key']}"
        )

        if uploaded_img_file is not None:
            try:
                with st.spinner(f"Extracting data from image '{uploaded_img_file.name}'..."):
                    img = Image.open(uploaded_img_file)
                    st.image(img, caption="Processing Image...", use_container_width=True)
                    
                    image_bytes = uploaded_img_file.getvalue()
                    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                    image_data_url = f"data:image/png;base64,{encoded_image}"
                    
                    vision_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                    
                    vision_prompt = """
                    Analyze this image carefully. If it is a receipt, invoice, or chart, extract the tabular data.
                    You MUST return ONLY a raw, valid JSON array of objects. Do not use markdown blocks like ```json.
                    Make sure the keys are clean, lowercase strings (e.g., 'item_name', 'price', 'date').
                    If you cannot find tabular data, return an empty array [].
                    """
                    
                    response = vision_llm.invoke([
                        {"role": "user", "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}}
                        ]}
                    ])
                    
                    raw_text = response.content.strip().strip('```json').strip('```')
                    extracted_data = json.loads(raw_text)
                    
                    if not extracted_data:
                        st.warning("⚠️ No tabular data was found in this image.")
                    else:
                        # Bulletproof JSON Parsing
                        if isinstance(extracted_data, dict):
                            first_key = list(extracted_data.keys())[0]
                            df = pd.DataFrame(extracted_data[first_key])
                        else:
                            df = pd.DataFrame(extracted_data)
                            
                        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower() for col in df.columns]
                        
                        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                        file_brief = re.sub(r'[^a-zA-Z0-9_]', '_', uploaded_img_file.name.rsplit('.', 1)[0].lower())[:15]
                        clean_name = f"table_{timestamp}_{file_brief}_img"
                        
                        engine = create_engine(databricks_uri, poolclass=NullPool)
                        df.to_sql(clean_name, con=engine, if_exists="replace", index=False, method=databricks_insert)
                        
                        st.session_state["last_uploaded_img"] = clean_name
                        get_visual_agent.clear() 
                        st.session_state["uploader_key"] += 1
                        st.rerun()

            except json.JSONDecodeError:
                st.error("🛑 Vision Error: The AI could not format the image data into a clean table.")
            except Exception as e:
                st.error(f"🛑 File Error: Could not process this image. Details: {e}")

    # --- Persistent Memory Display ---
    st.divider()
    st.markdown("##### 🧠 Active System Memory")
    if "last_uploaded_sql" in st.session_state:
        st.success(f"📊 DB: `{st.session_state['last_uploaded_sql']}`")
    if "last_uploaded_pdf" in st.session_state:
        st.success(f"📄 RAG: `{st.session_state['last_uploaded_pdf']}`")
    if "last_uploaded_img" in st.session_state:
        st.success(f"👁️ Vision: `{st.session_state['last_uploaded_img']}`")

# --- 5. Instantiate Agent ---
agent, chart_holder = get_visual_agent()
config = {"configurable": {"thread_id": "user_session_1"}}

# --- 6. Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "chart" in message and message["chart"] is not None:
            st.plotly_chart(message["chart"], use_container_width=True)
        st.markdown(message["content"])

# --- 7. Execution Loop & HITL Security ---
if "pending_action" in st.session_state and st.session_state["pending_action"]:
    action = st.session_state["pending_action"]
    with st.chat_message("assistant"):
        st.warning("🚨 **Sensitive Action Detected!**")
        st.markdown(f"**Action Requested:**\n{action['display_msg']}")
        col1, col2 = st.columns(2)
        if col1.button("✅ Approve & Run"):
            try:
                engine = create_engine(databricks_uri, poolclass=NullPool)
                with engine.begin() as conn: 
                    conn.execute(text(action['sql']))
                get_visual_agent.clear() 
                success_msg = "✅ **Confirmation received. Action executed, and memory refreshed.**"
                st.balloons(); st.success(success_msg)
                st.session_state.messages.append({"role": "assistant", "content": success_msg})
                st.session_state["pending_action"] = None
                st.rerun() 
            except Exception as e:
                st.error(f"Execution Error: {e}")
                st.session_state["pending_action"] = None
        if col2.button("❌ Reject"):
            st.info("Action cancelled."); st.session_state["pending_action"] = None; st.rerun()

if prompt := st.chat_input("E.g., Based on the PDF, why did costs rise? Draw a chart of actual vs budget."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.container()
        with st.spinner("Synthesizing data, writing code, and self-correcting..."):
            try:
                chart_holder.clear()
                
                lg_messages = []
                if len(st.session_state.messages) > 1:
                    for msg in st.session_state.messages[-5:-1]:
                        lg_messages.append((msg["role"], msg["content"]))
                
                lg_messages.append(("user", prompt))

                # Retry logic for 500 Server Errors
                response = None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = agent.invoke({"messages": lg_messages}, config=config)
                        break
                    except Exception as e:
                        if ("ServerError" in str(e) or "500" in str(e)) and attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            raise e
                
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

                # Bulletproof Security Gate Regex
                if "SECURITY_CONFIRMATION_REQUIRED" in final_text:
                    tick3 = '```'
                    pattern = tick3 + r"(?:sql)?\n?(.*?)\n?" + tick3
                    sql_match = re.search(pattern, final_text, re.DOTALL | re.IGNORECASE)
                    
                    clean_msg = final_text.replace("SECURITY_CONFIRMATION_REQUIRED", "").strip()
                    if sql_match:
                        st.session_state["pending_action"] = {"sql": sql_match.group(1), "display_msg": clean_msg}
                        st.session_state.messages.append({"role": "assistant", "content": clean_msg})
                        st.rerun()
                    else:
                        err_msg = "🚨 **Sensitive Action Detected!** The AI requested an action but forgot to format the SQL command in a code block."
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                
                else:
                    generated_chart = chart_holder.get('current_fig', None) 
                    if not final_text.strip():
                        if generated_chart:
                            final_text = "I successfully generated the chart based on the data."
                        else:
                            final_text = "⚠️ The agent hit a dead end and returned no text. Check the thought process below to see where it failed."
                    
                    if generated_chart:
                        st.plotly_chart(generated_chart, use_container_width=True)
                    
                    st.markdown(final_text)
                    
                    with st.expander("🔍 View Agent's Internal Thought Process"):
                        for i, msg in enumerate(response["messages"]):
                            if getattr(msg, 'type', '') == 'system' or i == 0: 
                                continue 
                            
                            st.markdown(f"**Step {i} ({getattr(msg, 'type', 'unknown')})**:")
                            
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                st.write("🔧 **Triggered Tools:**")
                                for tool in msg.tool_calls:
                                    st.code(f"{tool['name']}: {tool['args']}")
                            
                            if msg.content:
                                if "```" in str(msg.content):
                                    st.markdown(msg.content)
                                else:
                                    st.code(str(msg.content)[:1000] + ("..." if len(str(msg.content)) > 1000 else ""))
                            
                            st.divider()
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_text,
                        "chart": generated_chart
                    })
                    
            except Exception as e:
                st.error(f"Error: {e}")

# --- requirements.txt ---
# streamlit
# pandas==2.1.4
# sqlalchemy
# plotly
# langchain
# langchain-core
# langchain-community
# langchain-google-genai
# langchain-experimental
# langchain-text-splitters
# databricks-sql-connector
# databricks-sqlalchemy
# openpyxl
# langgraph
# pypdf
# faiss-cpu
# sentence-transformers
# pillow