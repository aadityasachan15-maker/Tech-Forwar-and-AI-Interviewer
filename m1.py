from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv


from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


app = FastAPI()


# CORS allow necessary
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY .env file set")

client = genai.Client(api_key=API_KEY)


# Model Name - Gemma 4 latest variant
MODEL_ID = "gemma-4-26b-a4b-it"


class ChatRequest(BaseModel):
    prompt: str
    system_instruction: str = "You are a helpful AI assistant."


@app.get("/")
async def root():
    return {"status": "ok", "message": "Backend chal raha hai. Docs ke liye /docs kholo."}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=request.system_instruction
            ),
            contents=request.prompt,
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/think")
async def chat_with_thinking(request: ChatRequest):
    """Use it to solve complex logic problems and reasoning tasks."""
    try:
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=request.system_instruction,
                thinking_config=types.ThinkingConfig(thinking_level="high"),
            ),
            contents=request.prompt,
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__m1__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)












































