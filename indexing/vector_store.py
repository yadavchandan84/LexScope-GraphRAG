from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
)

from config import settings

COLLECTION_NAME = "legal_chunks"
DENSE_VECTOR_SIZE = 1024  # bge-large-en-v1.5 output dimension


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    # Qdrant Cloud when a URL is configured, otherwise an on-disk embedded instance
    # (no server process). Embedded mode locks the folder to one client per process,
    # so we cache a single instance and reuse it across requests.
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantClient(path=settings.qdrant_local_path)


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
