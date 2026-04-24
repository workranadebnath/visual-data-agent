🛡️ Visual Data Worker: Enterprise Edition
Visual Data Worker is an autonomous, omni-modal AI Data Agent that acts as a secure, senior-level data analyst. It bridges the gap between raw enterprise data and non-technical stakeholders by combining SQL querying, Retrieval-Augmented Generation (RAG), computer vision, and automated data visualization into a single, conversational interface.

Developed by Rana Debnath.

✨ Enterprise Features
📂 Omni-Channel Data Ingestion
Structured Data (CSV/Excel): Automated ETL pipeline that cleans column names, generates timestamped smart-names (tbl_YYYY_MM_DD_HH_MM_SS_filename), and securely pushes data to Databricks.

Qualitative Data (PDF RAG): Vectorizes PDF reports using FAISS and all-MiniLM-L6-v2 embeddings, allowing the agent to combine quantitative database metrics with qualitative business strategies.

Computer Vision (Images/Receipts): Utilizes Gemini Vision to extract tabular data from images and receipts. Features a custom bulletproof JSON parser to handle unpredictable LLM formatting dynamically.

🧠 Advanced LangGraph Cognitive Engine
Self-Healing Code Execution: Built on a StateGraph, the agent writes its own Python and Spark SQL code. If it encounters an error, it reads the traceback, self-corrects the code, and reruns the tool autonomously.

Persistent Active Memory: Tracks dynamically generated SQL tables, RAG documents, and Vision tables across the user session to prevent schema bloat and hallucination.

📈 Autonomous Visualizations
The agent is strictly instructed to use an isolated PythonAstREPLTool sandbox to write and execute pandas and plotly.express code.

It securely renders generated charts directly into the Streamlit UI via a globally injected memory state, ensuring the AI cannot crash the app with rogue UI commands.

🔐 "Human-in-the-Loop" (HITL) Security Gate
Destructive Action Interception: The agent is strictly forbidden from executing INSERT, UPDATE, DELETE, or DROP commands autonomously.

If a user requests a data modification, the AI writes the SQL, halts execution, and triggers a custom Streamlit UI roadblock. The command is only executed against Databricks if the human clicks ✅ Approve & Run.

🏗️ Architecture & Robustness
Databricks Connection Stability: Engineered with SQLAlchemy's NullPool architecture to bypass Databricks session timeouts and stale connection drops, ensuring 100% uptime for queries.

Spark SQL Dialect Enforcement: The prompt architecture forces the LLM to strictly adhere to Databricks Spark SQL syntax (e.g., bypassing SQLite limitations).

Regex State-Sync: Custom regex patterns ensure secure extraction of SQL commands from the agent's internal thought process without breaking Streamlit's UI rendering loop.

🚀 Installation & Setup
1. Clone the Repository
Bash
git clone https://github.com/yourusername/visual-data-worker.git
cd visual-data-worker
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Configure Secrets
Create a .streamlit folder in the root directory and add a secrets.toml file. Provide your Databricks and Google API credentials:

Ini, TOML
# .streamlit/secrets.toml

DATABRICKS_HOST = "your-databricks-workspace.cloud.databricks.com"
DATABRICKS_HTTP_PATH = "/sql/1.0/endpoints/your-endpoint-id"
DATABRICKS_TOKEN = "dapi-your-secure-databricks-token"

GOOGLE_API_KEY = "AIza-your-gemini-api-key"
(Note: Ensure your Databricks catalog and schema match the ones defined in the databricks_uri variable in app.py, or update the code to reflect your schema).

4. Run the Application
Bash
streamlit run app.py
🛠️ Tech Stack
Frontend: Streamlit

Data Warehouse: Databricks (Spark SQL)

Agent Framework: LangChain, LangGraph

LLMs: Google Gemini 2.5 Flash, Gemini Vision

Embeddings & Vector DB: HuggingFace (all-MiniLM-L6-v2), FAISS

Visualization & Data Parsing: Plotly, Pandas, SQLAlchemy

💡 Usage Examples
"Draw a pie chart of the top 10 regions by actual amount." -> Agent writes SQL, passes it to the Python Sandbox, and renders a Plotly chart.

"Based on the Q3 Report PDF, why did our marketing costs rise?" -> Agent queries the FAISS vector store and returns qualitative context.

"Delete the Netflix Top 10 table." -> Agent drafts the DROP TABLE command and triggers the Red Security Warning for human approval.
