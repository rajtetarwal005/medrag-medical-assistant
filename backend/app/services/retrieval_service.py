from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.schema import Document

from app.core.config import settings


# ---------------------------------------------------------
# Embedding Model (Shared)
# ---------------------------------------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL
    )


# ---------------------------------------------------------
# Load FAISS Vector Store
# ---------------------------------------------------------
def load_vectorstore():
    embedding_model = get_embedding_model()

    db = FAISS.load_local(
        settings.DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# ---------------------------------------------------------
# Basic Vector Retriever
# ---------------------------------------------------------
def get_vector_retriever(k: int = 20):
    db = load_vectorstore()

    return db.as_retriever(
        search_kwargs={"k": k}
    )


# ---------------------------------------------------------
# Hybrid Retriever (Vector + BM25)
# ---------------------------------------------------------
def get_hybrid_retriever(k_vector: int = 20, k_bm25: int = 20):
    db = load_vectorstore()

    # Vector Retriever
    vector_retriever = db.as_retriever(
        search_kwargs={"k": k_vector}
    )

    # Prepare documents for BM25
    all_docs = db.docstore._dict.values()
    all_docs = [doc for doc in all_docs if isinstance(doc, Document)]

    # BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k_bm25

    return vector_retriever, bm25_retriever