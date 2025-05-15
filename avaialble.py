import google.generativeai as genai

# Use your Gemini API key here
api_key = "AIzaSyBhWwQ_sgZymk0uRMjwWRkahaHqBAunGz4"

genai.configure(api_key=api_key)

# Print available models and their generation methods
try:
    print("Available Gemini models:\n")
    for m in genai.list_models():
        print(f"- {m.name} (generation methods: {m.supported_generation_methods})")
except Exception as e:
    print(f"Error fetching models: {e}")
