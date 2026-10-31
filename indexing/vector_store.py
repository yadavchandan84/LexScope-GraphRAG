from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
)

from config import settings

COLLECTION_NAME = "legal_chunks"
# Must match the configured embedding model's output size (default bge-small=384).
DENSE_VECTOR_SIZE = settings.embedding_dim


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    # Qdrant Cloud when a URL is configured, otherwise an on-disk embedded instance
    # (no server process). Embedded mode locks the folder to one client per process,
    # so we cache a single instance and reuse it across requests.
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantClient(path=settings.qdrant_local_path)


def _existing_dense_size(client: QdrantClient) -> int | None:
    info = client.get_collection(COLLECTION_NAME)
    vectors = info.config.params.vectors
    dense = vectors.get("dense") if isinstance(vectors, dict) else None
    return dense.size if dense is not None else None


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        existing = _existing_dense_size(client)
        if existing == DENSE_VECTOR_SIZE:
            return
        # The embedding model changed (e.g. bge-large 1024 -> bge-small 384).
        # Old vectors are unusable with the new model, so rebuild the collection.
        print(
            f"Collection '{COLLECTION_NAME}' has dense size {existing}, expected "
            f"{DENSE_VECTOR_SIZE}. Recreating it for the new embedding model."
        )
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
