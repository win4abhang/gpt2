from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

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
