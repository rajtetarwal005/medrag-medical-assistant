from fastapi import APIRouter
from pydantic import BaseModel   
from app.schemas.schema import QueryRequest, QueryResponse
from app.services.ingestion_service import run_ingestion
from app.services.rag_service import run_rag

router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    answer, docs = run_rag(
        request.query,
        session_id=request.session_id
    )

    # Handle sources safely
    sources = list(set(
        doc.metadata.get("source", "unknown") for doc in docs
    )) if docs else []

    return QueryResponse(
        answer=answer,
        sources=sources
    )


class UploadRequest(BaseModel):
    file_path: str


@router.post("/upload")
def upload_file(request: UploadRequest):
    run_ingestion(request.file_path)
    return {"message": "File processed successfully"}