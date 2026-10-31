from functools import lru_cache

from fastembed import TextEmbedding

from caching import cache_get_json, cache_set_json, make_key
from config import settings

# Dense embeddings via fastembed (ONNX Runtime, no PyTorch). The same function
# is used for indexing and for queries, so both sides always match.
_MODEL_NAME = settings.embedding_model
# bge models are trained with this instruction prefix on the query side.
_QUERY_PREFIX = "Represent this question for retrieval: "
_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    # threads=None lets ONNX use every core (fast local indexing). On small hosts
    # set ONNX_THREADS=1 to keep per-thread buffers and RAM down.
    return TextEmbedding(
        model_name=_MODEL_NAME,
        cache_dir=settings.model_cache_dir,
        threads=settings.onnx_threads,
    )


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return [vec.tolist() for vec in model.embed(texts, batch_size=_BATCH_SIZE)]


def embed_query(text: str) -> list[float]:
    # Query embeddings are deterministic per model, so cache them: a repeated
    # question skips the embedding step entirely.
    key = make_key("dense", _MODEL_NAME, text)
    cached = cache_get_json(key)
    if cached is not None:
        return cached

    model = get_embedder()
    vector = next(iter(model.embed([_QUERY_PREFIX + text]))).tolist()
    cache_set_json(key, vector, ttl=settings.cache_embedding_ttl)
    return vector
