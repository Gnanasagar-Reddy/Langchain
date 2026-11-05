from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
# 1. Define the output schema
class JokeSchema(BaseModel):
    setup: str = Field(description="The setup for the joke")
    punchline: str = Field(description="The punchline for the joke")

# 2. Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# 3. Enable structured output
structured_llm = llm.with_structured_output(JokeSchema)

# 4. Invoke the model
result = structured_llm.invoke("Tell me a joke")

# 5. Access the structured response
print("Setup:", result.setup)
print("Punchline:", result.punchline)

# Convert to dictionary
print(result.model_dump())
