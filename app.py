from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# Load the GPT-2 model
generator = pipeline("text-generation", model="distilgpt2")

# Set up FastAPI app
app = FastAPI()

class TextInput(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
def generate_text(input: TextInput):
    result = generator(input.prompt, max_length=input.max_length, num_return_sequences=1)
    return {"generated_text": result[0]["generated_text"]}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000)