
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

str_parser = StrOutputParser()

class SentimentResult(BaseModel):
    sentiment: Literal["Positive", "Negative"]

sentiment_llm = llm.with_structured_output(SentimentResult)

sentiment_prompt = PromptTemplate.from_template("""
Analyze the sentiment of the following movie review.

Review: {review}
""")

sentiment_chain = sentiment_prompt | sentiment_llm

positive_prompt = PromptTemplate.from_template("""
The following movie review is positive.

Review: {review}

Summarize the review, highlighting what makes the movie good.
""")

positive_chain = positive_prompt | llm | str_parser

negative_prompt = PromptTemplate.from_template("""
The following movie review is negative.

Review: {review}

Summarize the review, highlighting the criticisms and problems.
""")

negative_chain = negative_prompt | llm | str_parser

def prepare_input(data: dict):
    review = data["review"]

    sentiment_result = sentiment_chain.invoke({"review": review})
    sentiment = sentiment_result.sentiment

    return {
        "review": review,
        "sentiment": sentiment
    }

prepare_runnable = RunnableLambda(prepare_input)

conditional_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "Positive", positive_chain),
    negative_chain
)

def generate_summary(data: dict):
    summary = conditional_chain.invoke(data)

    return {
        "review": data["review"],
        "sentiment": data["sentiment"],
        "summary": summary
    }

summary_runnable = RunnableLambda(generate_summary)

def beautify(response: dict):
    return f"""
╔══════════════════════════════════════════════════╗
║           MOVIE REVIEW ANALYSIS               ║
╚══════════════════════════════════════════════════╝

🎬 Sentiment: {response["sentiment"]}

📝 Review Summary:
{response["summary"]}
"""

beautify_runnable = RunnableLambda(beautify)

final_chain = (
    prepare_runnable
    | summary_runnable
    | beautify_runnable
)

review = """
The movie was absolutely fantastic! The actors delivered incredible performances.
The screenplay was engaging, the visuals were stunning, and the music perfectly
complemented every scene. One of the best movies I have ever watched.
"""

result = final_chain.invoke({"review": review})

print(result)
