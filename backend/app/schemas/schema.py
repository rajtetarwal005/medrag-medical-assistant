from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class UploadRequest(BaseModel):
    file_path: str
    session_id: str    