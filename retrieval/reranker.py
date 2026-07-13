from functools import lru_cache

from sentence_transformers import CrossEncoder

from retrieval.hybrid_retriever import RetrievedChunk

_MODEL_NAME = "BAAI/bge-reranker-large"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(_MODEL_NAME)


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int = 8) -> list[RetrievedChunk]:
    if not chunks:
        return []

    model = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked[:top_k]
