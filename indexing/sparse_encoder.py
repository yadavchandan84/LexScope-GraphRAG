from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

_MODEL_NAME = "Qdrant/bm25"


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=_MODEL_NAME)


def sparse_embed_documents(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    embeddings = list(model.embed(texts))
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in embeddings
    ]


def sparse_embed_query(text: str) -> SparseVector:
    return sparse_embed_documents([text])[0]
