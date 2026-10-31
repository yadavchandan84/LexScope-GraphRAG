from functools import lru_cache

from sentence_transformers import SentenceTransformer

from caching import cache_get_json, cache_set_json, make_key
from config import settings

# Model name comes from config (default: BAAI/bge-small-en-v1.5). bge models are
# trained with this query instruction prefix for retrieval.
_MODEL_NAME = settings.embedding_model
_QUERY_PREFIX = "Represent this question for retrieval: "


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    # Query embeddings are deterministic per model, so cache them: a repeated
    # question skips the embedding forward pass entirely.
    key = make_key("dense", _MODEL_NAME, text)
    cached = cache_get_json(key)
    if cached is not None:
        return cached

    model = get_embedder()
    vector = model.encode(_QUERY_PREFIX + text, normalize_embeddings=True).tolist()
    cache_set_json(key, vector, ttl=settings.cache_embedding_ttl)
    return vector
