import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

llm_genai = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",

)

response = llm_genai.invoke("Bro, tell me a fun fact")
print(response.content)