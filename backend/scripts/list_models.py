import os
from google import genai
from dotenv import load_dotenv

# Load env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

print("[*] Inspecting RAW attributes of the first Gemini model...")
try:
    for model in client.models.list():
        if "gemini" in model.name.lower():
            print(f" --- Model: {model.name} ---")
            # In ra tất cả các thuộc tính của object model
            for attr in dir(model):
                if not attr.startswith('_'):
                    try:
                        val = getattr(model, attr)
                        print(f"  {attr}: {val}")
                    except:
                        pass
            break # Chỉ cần xem 1 model để biết cấu trúc
except Exception as e:
    print(f"❌ Error during inspection: {e}")
