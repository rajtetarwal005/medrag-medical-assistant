import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    DATA_PATH: str = os.getenv("DATA_PATH", "data/")
    DB_FAISS_PATH: str = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )


settings = Settings()