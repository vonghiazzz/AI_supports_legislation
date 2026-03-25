import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

print("Listing models...")
print("Listing Gemini models with generateContent...")
for model in client.models.list():
    if "gemini" in model.name.lower() and "generateContent" in model.supported_actions:
        print(f"Model ID: {model.name}")
