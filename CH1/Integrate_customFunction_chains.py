import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence

# STEP 1: Load environment variables
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# STEP 2: Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
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

# STEP 4: Convert text into dictionary
def text_to_dict(text):
    result = {}

    for line in text.strip().splitlines():
        key, value = line.split(":", 1)
        result[key.strip().lower()] = value.strip()

    return result

dict_runnable = RunnableLambda(text_to_dict)

# STEP 5: LinkedIn prompt
linkedin_prompt = PromptTemplate.from_template(
    """
    Create an engaging LinkedIn post about the capital of {country}.

    Capital: {capital}

    Include:
    - An interesting introduction
    - Interesting facts
    - Professional tone
    - Relevant hashtags
    """
)

# STEP 6: Complete sequential chain
chain = RunnableSequence(
    capital_prompt,
    llm,
    str_parser,
    dict_runnable,
    linkedin_prompt,
    llm,
    str_parser
)

# STEP 7: Execute
result = chain.invoke({"country": "France"})

print(result)