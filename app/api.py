from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import QueryRequest, QueryResponse
from generation.generate import generate_answer
from retrieval.cli import run_retrieval_pipeline

app = FastAPI(title="Legal RAG API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    chunks = run_retrieval_pipeline(req.query, doc_type_filter=req.doc_type_filter, rerank_top_k=req.top_k)
    result = generate_answer(req.query, chunks)
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
        insufficient_context=result.insufficient_context,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
