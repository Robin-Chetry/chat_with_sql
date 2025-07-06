import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
import sqlite3

# --- App Setup ---
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
radio_opt = ["Use SQLLite 3 Database- Student.db", "Connect to you MySQL Database"]
selected_opt = st.sidebar.radio("Choose the DB which you want to chat", options=radio_opt)

# --- Sidebar Inputs ---
if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input("Groq API Key", type="password")
default_greeting = st.sidebar.text_input("Customize Assistant Greeting", "How can I help you?")
clear_button = st.sidebar.button("Clear message history")

# --- Input Validations ---
if not api_key:
    st.warning("Please provide your Groq API Key.")
    st.stop()

if db_uri == MYSQL and not all([mysql_host, mysql_user, mysql_password, mysql_db]):
    st.warning("Please fill all MySQL fields.")
    st.stop()

# --- LLM Setup ---
@st.cache_resource(ttl="2h")
def setup_llm(api_key):
    return ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

# --- Database Setup ---
@st.cache_resource(ttl="2h")
def setup_db():
    if db_uri == LOCALDB:
        db_path = Path(__file__).parent / "student.db"
        if not db_path.exists():
            st.error("SQLite file 'student.db' not found!")
            st.stop()
        creator = lambda: sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    else:
        return SQLDatabase(
            create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")
        )

# --- Agent Setup ---
def setup_agent(llm, db):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(llm=llm, toolkit=toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# --- Load Components ---
llm = setup_llm(api_key)
db = setup_db()
agent = setup_agent(llm, db)

# --- Chat Session Setup ---
if "messages" not in st.session_state or clear_button:
    st.session_state["messages"] = [{"role": "assistant", "content": default_greeting}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# --- User Input and Agent Run ---
if user_query := st.chat_input("Ask anything from the database"):
    user_query = user_query.strip()
    if user_query == "":
        st.warning("Empty query entered!")
    else:
        st.session_state["messages"].append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    callback = StreamlitCallbackHandler(st.container())
                    response = agent.run(user_query, callbacks=[callback])
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.write(response)
        except Exception as e:
            st.error("Error during processing.")
            st.exception(e)
