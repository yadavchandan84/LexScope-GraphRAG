from functools import lru_cache

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_QUERY_PREFIX = "Represent this question for retrieval: "


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    model = get_embedder()
    return model.encode(_QUERY_PREFIX + text, normalize_embeddings=True).tolist()
