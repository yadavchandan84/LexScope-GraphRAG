from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    doc_type_filter: str | None = None
    top_k: int = 8


class Citation(BaseModel):
    doc_name: str
    page: int
    excerpt_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str
    insufficient_context: bool
    retrieved_chunk_ids: list[str]
