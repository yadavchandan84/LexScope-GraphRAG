from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import QueryRequest, QueryResponse
from app.service import answer_query
from caching import ping as cache_ping

app = FastAPI(title="Legal RAG API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
async def _warm_up() -> None:
    # Preload the heavy local models (embedder, sparse encoder, reranker) so the
    # first real request doesn't pay the cold-start cost. Best effort — never
    # block startup if a model or service is briefly unavailable.
    try:
        await run_in_threadpool(_preload_models)
    except Exception:
        pass


def _preload_models() -> None:
    from config import settings
    from indexing.embedder import get_embedder
    from indexing.sparse_encoder import get_sparse_model

    get_embedder()
    get_sparse_model()
    if settings.reranker_enabled:
        from retrieval.reranker import get_reranker

        get_reranker()


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    # The pipeline is blocking (CPU-bound models + network I/O). Run it in a
    # worker thread so it never blocks the event loop / other requests.
    payload = await run_in_threadpool(
        answer_query,
        req.query,
        req.doc_type_filter,
        req.top_k,
    )
    return QueryResponse(**payload)


@app.get("/health")
async def health():
    return {"status": "ok", "cache": cache_ping()}
