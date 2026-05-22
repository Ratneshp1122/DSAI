"""
check_models.py - Lists all models available with your API key.
Run: py check_models.py
"""
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "")

if not api_key:
    print("❌ No GEMINI_API_KEY in .env")
    exit(1)

print(f"Using key: {api_key[:12]}...{api_key[-4:]}\n")

try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    print("=== Models supporting generateContent ===")
    count = 0
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  ✅ {m.name}")
            count += 1
    print(f"\nTotal: {count} models available")
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying google-genai SDK...")
    try:
        from google import genai as genai2
        client = genai2.Client(api_key=api_key)
        for m in client.models.list():
            print(f"  {m.name}")
    except Exception as e2:
        print(f"Also failed: {e2}")
