
from pathlib import Path
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing")

print("Creating client...", flush=True)
client = genai.Client(api_key=api_key)

print("Sending request to Gemini...", flush=True)
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="reply with 20 words greeting"
)

print("Gemini response:", response.text, flush=True)
