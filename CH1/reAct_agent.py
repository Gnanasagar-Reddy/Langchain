
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ddg = DuckDuckGoSearchRun()

@tool
def search_tool(query: str) -> str:
    """Search the web for recent news, current events, and general information."""
    try:
        return ddg.invoke(query)
    except Exception as e:
        return f"Search temporarily failed: {str(e)}. Try Wikipedia or answer using available information."

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

@tool
def enterprise_tool(query: str) -> str:
    """Search for enterprise and business-related information."""
    return f"Enterprise search results for: {query}"

ToolKit = [
    search_tool,
    wikipedia_tool,
    enterprise_tool
]

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_tokens=1000,
    timeout=30
)

agent = create_agent(
    model=model,
    tools=ToolKit,
    system_prompt="""
You are a helpful research assistant.
Use web search for current and latest information.
Use Wikipedia for general factual information.
Do not make repeated searches for the same information.
Use at most 2 search calls per question.
If a tool fails, continue using available information.
After gathering sufficient information, provide a clear final answer.
"""
)

example_query = "Give me the latest news about the stock market."

events = agent.stream(
    {"messages": [("user", example_query)]},
    config={"recursion_limit": 10},
    stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()
