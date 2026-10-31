from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from caching import cache_get_json, cache_set_json, make_key
from config import settings

_MODEL_NAME = "Qdrant/bm25"


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(
        model_name=_MODEL_NAME,
        cache_dir=settings.model_cache_dir,
        threads=settings.onnx_threads,
    )


def sparse_embed_documents(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    embeddings = list(model.embed(texts))
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in embeddings
    ]


def sparse_embed_query(text: str) -> SparseVector:
    # Cache the sparse (BM25) query vector too, mirroring the dense path.
    key = make_key("sparse", _MODEL_NAME, text)
    cached = cache_get_json(key)
    if cached is not None:
        return SparseVector(indices=cached["indices"], values=cached["values"])

    vector = sparse_embed_documents([text])[0]
    cache_set_json(
        key,
        {"indices": vector.indices, "values": vector.values},
        ttl=settings.cache_embedding_ttl,
    )
    return vector
