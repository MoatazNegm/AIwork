
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

# Initialize FastAPI
app = FastAPI()

# Configuration - set your API key here or via environment variable
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-2954077ef6724ad5a51b6dcdb2ad07fe")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # Example endpoint, adjust if different

# Define request/response models
class Message(BaseModel):
    role: str
    content: str

class InferenceRequest(BaseModel):
    messages: list[Message]  # DeepSeek API typically uses a chat format
    max_tokens: int = 100
    model: str = "deepseek-chat"  # or whatever model name DeepSeek provides

class InferenceResponse(BaseModel):
    generated_text: str

# Inference endpoint
@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": request.model,
        "messages": [msg.dict() for msg in request.messages],
        "max_tokens": request.max_tokens
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract the generated text from the response
        # Adjust this based on the actual API response structure
        generated_text = result["choices"][0]["message"]["content"]

        return {"generated_text": generated_text}
    except requests.exceptions.RequestException as e:
        return {"generated_text": f"Error calling DeepSeek API: {str(e)}"}

# Run with: uvicorn main:app --reload
