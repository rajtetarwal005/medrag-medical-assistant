# Singleton model (load once)
_reranker_model = None


# Singleton model (load once)
_reranker_model = None


def get_reranker():
    global _reranker_model

    if _reranker_model is None:
        # ✅ IMPORT INSIDE FUNCTION
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return _reranker_model


def rerank_documents(query, docs, top_n: int = 4):
    """
    Rerank documents using cross-encoder model with threshold filtering.
    """

    if not docs:
        return []

    reranker = get_reranker()

    # Create (query, doc) pairs
    pairs = [(query, doc.page_content) for doc in docs]

    # Get scores
    scores = reranker.predict(pairs)

    # Apply threshold filtering
    THRESHOLD = 0.5  # you can tune this

    filtered_docs = [
        (doc, score) for doc, score in zip(docs, scores) if score > THRESHOLD
    ]

    # If nothing passes threshold
    if not filtered_docs:
        return []

    # Sort by score (descending)
    filtered_docs.sort(key=lambda x: x[1], reverse=True)

    # Return top N docs
    top_docs = [doc for doc, _ in filtered_docs[:top_n]]

    return top_docs