import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableSequence,
    RunnableParallel
)

# STEP 1: Load environment variables
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# STEP 2: Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

str_parser = StrOutputParser()


# STEP 3: Capital information prompt
capital_prompt = PromptTemplate.from_template(
    """
    Tell me the capital of {country}.

    Return exactly this format:
    Country: <country>
    Capital: <capital>
    """
)


# STEP 4: Custom function to convert text into dictionary
def text_to_dict(text):
    result = {}

    for line in text.strip().splitlines():
        key, value = line.split(":", 1)
        result[key.strip().lower()] = value.strip()

    return result


dictionary_maker_runnable = RunnableLambda(text_to_dict)


# STEP 5: LinkedIn prompt
linkedin_prompt = PromptTemplate.from_template(
    """
    Create a professional LinkedIn post about the capital of {country}.

    Capital: {capital}

    Include interesting facts and relevant hashtags.
    """
)


# STEP 6: LinkedIn chain (direct Runnable chain)
chain_linkedin = (
    linkedin_prompt
    | llm
    | str_parser
)


# STEP 7: Instagram chain inside a custom function
def insta_chain(text: dict):

    country = text["country"]
    capital = text["capital"]

    # TASK 1: Prompt
    insta_prompt = PromptTemplate.from_template(
        """
        Create an attractive Instagram post about {capital},
        the capital of {country}.

        Include:
        - Catchy caption
        - Emojis
        - Interesting facts
        - Relevant hashtags
        """
    )

    # TASK 2: LLM
    chain_insta = (
        insta_prompt
        | llm
        | str_parser
    )

    # Execute chain
    result = chain_insta.invoke({
        "country": country,
        "capital": capital
    })

    return result


# Convert custom function into Runnable
insta_chain_runnable = RunnableLambda(insta_chain)


# STEP 8: Parallel chains
parallel_chain = RunnableParallel(
    {
        "linkedin": chain_linkedin,
        "instagram": insta_chain_runnable
    }
)


# STEP 9: Final orchestration
final_chain = (
    capital_prompt
    | llm
    | str_parser
    | dictionary_maker_runnable
    | parallel_chain
)


def beautify(final_response):
    linkedin_post = final_response["linkedin"]
    instagram_post = final_response["instagram"]

    return f"""
╔══════════════════════════════════════════════════╗
║              LINKEDIN POST                     ║
╚══════════════════════════════════════════════════╝

{linkedin_post}


╔══════════════════════════════════════════════════╗
║              INSTAGRAM POST                     ║
╚══════════════════════════════════════════════════╝

{instagram_post}

════════════════════════════════════════════════════
"""


beautify_runnable = RunnableLambda(beautify)

final_chain = final_chain | beautify_runnable

result = final_chain.invoke({"country": "France"})

print(result)