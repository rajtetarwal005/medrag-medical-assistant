import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.services.retrieval_service import get_embedding_model

# ---------------------------------------------------------
# Load documents
# ---------------------------------------------------------
def load_documents():
    loader = DirectoryLoader(
        settings.DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


# ---------------------------------------------------------
# Split documents
# ---------------------------------------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL
    )


# ---------------------------------------------------------
# Build FAISS
# ---------------------------------------------------------
def build_vectorstore(chunks):
    embedding_model = get_embedding_model()

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(settings.DB_FAISS_PATH)

    return db


# ---------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------
def run_ingestion(file_path: str, session_id: str = "default"):

    # ---- Load PDF ----
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # ---- Split ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # ---- Add metadata (VERY IMPORTANT ) ----
    for chunk in chunks:
        chunk.metadata["source"] = file_path
        chunk.metadata["user_id"] = session_id

    embedding = get_embedding_model()

    DB_PATH = settings.DB_FAISS_PATH

    # ---- Load or create DB ----
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(
            DB_PATH,
            embedding,
            allow_dangerous_deserialization=True
        )
        db.add_documents(chunks)   # ✅ ADD NEW DATA
    else:
        db = FAISS.from_documents(chunks, embedding)

    # ---- Save DB ----
    db.save_local(DB_PATH)

    return {"message": "Ingestion completed"}

if __name__ == "__main__":
    run_ingestion()    

    # python app/services/ingestion_service.py to run the ingestion process standalone