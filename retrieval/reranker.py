from functools import lru_cache

from sentence_transformers import CrossEncoder

from config import settings
from retrieval.hybrid_retriever import RetrievedChunk

# Default BAAI/bge-reranker-base (~1.1GB) instead of -large (~2.2GB) for smaller
# hosts. Set RERANKER_ENABLED=false to skip the cross-encoder entirely and fall
# back to the hybrid-search (RRF) ordering — lowest memory footprint.
_MODEL_NAME = settings.reranker_model


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(_MODEL_NAME)


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int = 8) -> list[RetrievedChunk]:
    if not chunks:
        return []

    # When reranking is disabled, keep the hybrid-search order (already RRF-fused)
    # and just take the top_k. Saves loading the cross-encoder model.
    if not settings.reranker_enabled:
        return chunks[:top_k]

    model = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked[:top_k]
