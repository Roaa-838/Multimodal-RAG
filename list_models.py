import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=key)

print("Available models that support generateContent:\n")
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(f"  {m.name}")