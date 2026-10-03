import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

SIMULATE_RATE_LIMIT = os.getenv("SIMULATE_RATE_LIMIT", "0") == "1"

MODEL_NAMES = {
    "groq": "openai/gpt-oss-120b",
    "ollama": "llama3.2:3b",
}


def get_llm(provider):
    if provider == "groq":
        return ChatGroq(model=MODEL_NAMES["groq"], temperature=0.3, max_retries=1)
    return ChatOllama(model=MODEL_NAMES["ollama"], temperature=0.3)
