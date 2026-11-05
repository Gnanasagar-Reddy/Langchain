
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

str_parser = StrOutputParser()

print("\n--- Manual Invocation ---")

template = prompt_template.invoke({
    "input": "What is the capital of France?"
})

response = llm_gemini.invoke(template)

final_result = str_parser.invoke(response)

print(final_result)

print("\n--- Chain Invocation ---")

chain = prompt_template | llm_gemini | str_parser

result = chain.invoke({
    "input": "Explain what LangChain is in simple words."
})

print(result)
