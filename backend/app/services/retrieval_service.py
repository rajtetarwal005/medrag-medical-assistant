from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
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
# Hybrid Retriever (Vector + BM25) WITH USER FILTER
# ---------------------------------------------------------
def get_hybrid_retriever(session_id: str, k_vector: int = 20, k_bm25: int = 20):
    db = load_vectorstore()

    # 🔥 VECTOR RETRIEVER WITH FILTER
    vector_retriever = db.as_retriever(
        search_kwargs={
            "k": k_vector,
            "filter": {"user_id": session_id}   # ✅ IMPORTANT
        }
    )

    # 🔥 GET ALL DOCS
    all_docs = db.docstore._dict.values()
    all_docs = [doc for doc in all_docs if isinstance(doc, Document)]

    # 🔥 FILTER DOCS FOR BM25
    filtered_docs = [
        doc for doc in all_docs
        if doc.metadata.get("user_id") == session_id
    ]

    # 🔥 BM25 RETRIEVER (ONLY USER DATA)
    # 🔥 SAFE BM25 CREATION
    if len(filtered_docs) > 0:
        bm25_retriever = BM25Retriever.from_documents(filtered_docs)
        bm25_retriever.k = k_bm25
    else:
        bm25_retriever = None   

    return vector_retriever, bm25_retriever