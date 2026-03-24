from fastapi import APIRouter, UploadFile, File, Form
import os

from app.schemas.schema import QueryRequest, QueryResponse
from app.services.ingestion_service import run_ingestion
from app.services.rag_service import run_rag

router = APIRouter()


# -------------------- ASK API --------------------
@router.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    answer, docs = run_rag(
        request.query,
        session_id=request.session_id
    )

    sources = list(set(
        doc.metadata.get("source", "unknown") for doc in docs
    )) if docs else []

    return QueryResponse(
        answer=answer,
        sources=sources
    )


# -------------------- UPLOAD API (FIXED) --------------------
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    os.makedirs("uploaded_docs", exist_ok=True)

    file_path = f"uploaded_docs/{file.filename}"

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Run ingestion
    run_ingestion(file_path)

    return {"message": "File processed successfully"}