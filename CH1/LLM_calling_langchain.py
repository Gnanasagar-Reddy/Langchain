import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

llm_genai = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

message = [
    SystemMessage(content="Always respond in terms of the buissness or financial way !!"),
    HumanMessage(content="Bro,tell me a fun fact!")
]

response = llm_genai.invoke(message)
print(response.content)