import os
from langchain_groq import ChatGroq
from app.core.config import settings


def get_llm():
    """
    Initialize and return the Groq LLM.
    Centralized LLM configuration.
    """

    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=512,
        groq_api_key=settings.GROQ_API_KEY
    )