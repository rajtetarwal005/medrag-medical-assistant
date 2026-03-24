from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from app.utils.redis_memory import get_chat_history, save_chat_history

from app.services.retrieval_service import get_hybrid_retriever
from app.services.rerank_service import rerank_documents
from app.services.llm_service import get_llm


# ---------------------------------------------------------
# Prompt (with safety)
# ---------------------------------------------------------
CUSTOM_PROMPT = """
You are a medical assistant.

IMPORTANT RULES:
- Answer ONLY from the provided context
- Do NOT make up answers
- Do NOT give prescriptions
- Do NOT diagnose diseases
- If unsure, say "I don't know"
- Always recommend consulting a doctor if needed

Context:
{context}

Question:
{question}

Answer:
"""


def get_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT,
        input_variables=["context", "question"]
    )


# ---------------------------------------------------------
# Memory
# ---------------------------------------------------------
def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


# ---------------------------------------------------------
# Query Rewrite (optional but powerful)
# ---------------------------------------------------------
def rewrite_query(query, memory):
    if memory is None:
        return query

    history = memory.chat_memory.messages
    if not history:
        return query

    llm = get_llm()

    history_text = ""
    for msg in history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    rewrite_prompt = f"""
    You are given a conversation.

    Your job is to convert the latest question into a standalone question.

    IMPORTANT:
    - Replace words like "it", "this", "that" with actual subject
    - Keep full meaning
    - Do NOT change intent

    Conversation:
    {history_text}

    User Question:
    {query}

    Standalone Question:
    """

    try:
        response = llm.invoke(prompt)
        return response.content.strip() or query
    except:
        return query


# ---------------------------------------------------------
# MAIN RAG PIPELINE
# ---------------------------------------------------------
def run_rag(query: str, session_id: str = "default"):

    # ---- load history from Redis ----
    history = get_chat_history(session_id)

    # ---- convert history to text ----
    history_text = ""
    for item in history:
        history_text += f"User: {item['user']}\nAssistant: {item['assistant']}\n"

    # ---- rewrite query (SAFE VERSION) ----
    standalone_query = query

    if history_text:
        llm = get_llm()
        rewrite_prompt = f"""
You are given a conversation.

Convert the latest question into a standalone question.

IMPORTANT:
- Replace "it", "this", "that" with actual subject
- Keep meaning same

Conversation:
{history_text}

Question:
{query}

Standalone Question:
"""
        try:
            rewritten = llm.invoke(rewrite_prompt).content.strip()

            # 🔥 fallback protection
            if rewritten and len(rewritten) > 5:
                standalone_query = rewritten
            else:
                standalone_query = query

        except:
            standalone_query = query

    print("Original:", query)
    print("Rewritten:", standalone_query)

    # ---- Step 2: retrieval ----
    vector_ret, bm25_ret = get_hybrid_retriever()
    docs = vector_ret.invoke(standalone_query) + bm25_ret.invoke(standalone_query)

    # ---- deduplicate ----
    seen = set()
    unique_docs = []

    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    # ---- Step 3: rerank ----
    top_docs = rerank_documents(standalone_query, unique_docs, top_n=4)

    # ---- safety check ----
    if not top_docs:
        return "I don't know. No relevant medical information found.", []

    # ---- Step 4: build context ----
    context = "\n\n".join([doc.page_content for doc in top_docs])

    # ---- Step 5: LLM ----
    prompt = get_prompt().format(context=context, question=standalone_query)

    llm = get_llm()
    response = llm.invoke(prompt)

    # ---- Step 6: save history ----
    history.append({
        "user": query,
        "assistant": response.content
    })

    save_chat_history(session_id, history)

    return response.content, top_docs