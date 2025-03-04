from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

app = FastAPI(title="Mr. CatMan's Language Model")

# Create templates directory for HTML files
templates = Jinja2Templates(directory="templates")

# Cat-specific responses and behaviors
CAT_BEHAVIORS = [
    "purrs contentedly",
    "meows curiously",
    "stretches lazily",
    "tilts head inquisitively",
    "blinks slowly in approval",
    "paws at the screen playfully",
    "rolls over showing belly (it's a trap!)",
    "grooms whiskers thoughtfully"
]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate_response(request: Request):
    form_data = await request.form()
    user_input = form_data.get("input", "")
    
    # Process the input in a cat-friendly way
    response = process_cat_input(user_input)
    
    return {"response": response}

def process_cat_input(input_text: str) -> str:
    # Simple rule-based responses for now
    input_lower = input_text.lower()
    
    if "meow" in input_lower:
        return f"*{random.choice(CAT_BEHAVIORS)}* Meow meow! 😺"
    elif "food" in input_lower or "hungry" in input_lower:
        return "*paws excitedly at food bowl* Meooow! 🐱"
    elif "play" in input_lower:
        return "*bounces around energetically* Mrrrrp! 🐱‍👤"
    elif "sleep" in input_lower or "tired" in input_lower:
        return "*curls up into a cozy ball* Purrrrrr... 😴"
    else:
        return f"*{random.choice(CAT_BEHAVIORS)}* Mrow? 🐱" 