import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage


def connect_to_db(User : str , Password : str , Host : str , Port : int , Database : str):
    from_uri = f"mysql+mysqlconnector://{User}:{Password}@{Host}:{Port}/{Database}"
    return SQLDatabase.from_uri(from_uri)

def get_sql_chain(db):
    template =""" 
      You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
      Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
      Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    
      <SCHEMA>{schema}</SCHEMA>
    
      Conversation History: {chat_history}
    
      Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
      For example:
      Question: which 3 artists have the most tracks?
      SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
      Question: Name 10 artists
      SQL Query: SELECT Name FROM Artist LIMIT 10;
    
      Your turn:
      Question: {question}
      SQL Query:
      """
    
    prompt = ChatPromptTemplate.from_template(template)

    # Huggingface Model
    llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct")

    # Groq Model
    llm_groq = ChatGroq(model_name="llama-3.1-70b-versatile")

    def get_schema(_):
        return db.get_table_info()
    return (RunnablePassthrough.assign(schema = get_schema)
            | prompt
            | llm_groq
            | StrOutputParser()
            )

def get_response(user_query: str, db: SQLDatabase,chat_history: list):
    sql_chain = get_sql_chain(db)
    template = """
      You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
      Based on the table schema below, question, sql query, and sql response, write a natural language response.
      <SCHEMA>{schema}</SCHEMA>

      Conversation History: {chat_history}
      SQL Query: <SQL>{query}</SQL>
      User question: {question}
      SQL Response: {response}"""
    prompt = ChatPromptTemplate.from_template(template)
    
    llm_groq = ChatGroq(model_name="llama-3.1-70b-versatile")
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response= lambda vars: db.run(vars["query"])
        )
        | prompt
        | llm_groq
        | StrOutputParser()
        )
    return chain.invoke({"question": user_query, "chat_history": chat_history})



load_dotenv()

if "chat_histroy" not in st.session_state:
    st.session_state.chat_history =[
        AIMessage(content = "Hello i am here to help you. Ask me any thing !"),
        ]

st.set_page_config(page_title="Database Chat Bot", page_icon=":robot_face:", layout="centered")
st.title("Chat With Your Database")


with st.sidebar:
    st.subheader("Setting")
    st.write("This is simple Chat Bot")

    st.text_input("User" , value = "root" , key="User",help="Database Username")
    st.text_input("Password" , type = "password" ,value = "1999", key="Password",help="Database Password")
    st.text_input("Host", value = "localhost", key="Host",help="Database Host")
    st.text_input("Port",value = 3306, key="Port",help="Database Port")
    st.text_input("Database" , value = "sakila" , key= "Database",help="Database name")

    if st.button("Connect"):
        with st.spinner("Connecting to Database"):
            db = connect_to_db(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to Database")


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content = user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        #sql_chain = get_sql_chain(st.session_state.db)
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content = response))



