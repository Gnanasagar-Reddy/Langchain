import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_agent

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if os.environ.get("GEMINI_API_KEY"):
    print("Gemini API KEY Variable exists")
else:
    raise ValueError("GEMINI_API_KEY not found")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_tokens=1000,
    timeout=30
)

sql_db = SQLDatabase.from_uri("sqlite:///SalesDB/sales.db")

toolkit = SQLDatabaseToolkit(
    db=sql_db,
    llm=llm
)

tools = toolkit.get_tools()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a SQL database assistant. Use the available SQL tools to answer questions accurately. Inspect the database schema before writing queries. Execute read-only SQL queries and explain the results clearly."
)

example_query = """
Insert a new order into the orders table with these values:
customer_name = 'sagar'
product_name = 'TV'
quantity = 1
price = 100000
total = 100000

Execute the INSERT query and confirm whether it was successful.
"""

events = agent.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()