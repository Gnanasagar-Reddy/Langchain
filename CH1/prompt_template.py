
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

user_input=input("enter the topic you want the joke from :")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly science teacher. Explain simply and short 2 liner."),
    ("human", "Tell me one fun fact about {topic}.")
])

ready_prompt = prompt.invoke({"topic": user_input})
response = llm.invoke(ready_prompt)

print(response.content)
