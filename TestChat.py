import warnings
from langchain_groq import ChatGroq
from sqlalchemy import exc as sa_exc
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Suppress specific SQLAlchemy warnings
warnings.filterwarnings("ignore", category=sa_exc.SAWarning, message="Did not recognize type 'geometry'")
warnings.filterwarnings("ignore", category=sa_exc.SAWarning, message="Cannot correctly sort tables; there are unresolvable cycles")



def connect_to_db(User: str, Password: str, Host: str, Port: int, Database: str):
    # Create a SQLAlchemy engine with the correct URI
    database_uri = f"mysql+mysqlconnector://{User}:{Password}@{Host}:{Port}/{Database}"
    engine = create_engine(database_uri)

    # Reflect the database metadata, skipping the foreign key resolution warnings
    metadata = MetaData()
    metadata.reflect(bind=engine, resolve_fks=False)

    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()

    # Pass the engine and metadata to SQLDatabase
    return SQLDatabase(engine, metadata=metadata)

def get_sql_chain(db):
    template =""" 
      You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
      Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
      Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    
      <SCHEMA>{schema}</SCHEMA>
    
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

    # Groq Model
    llm_groq = ChatGroq(model_name="llama-3.1-70b-versatile")

    def get_schema(_):
        return db.get_table_info()
    return (RunnablePassthrough.assign(schema = get_schema)
            | prompt
            | llm_groq
            | StrOutputParser()
            )

def get_response(user_query: str, db: SQLDatabase):
    sql_chain = get_sql_chain(db)
    template = """
      You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
      Based on the table schema below, question, sql query, and sql response, write a natural language response.
      <SCHEMA>{schema}</SCHEMA>

    
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
    return chain.invoke({"question": user_query})

db = connect_to_db("root","1999","localhost",3306,"sakila")

user_query = "how many film are there"
response = get_response(user_query, db)
print(response)