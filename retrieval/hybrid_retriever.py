from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from indexing.embedder import embed_query
from indexing.sparse_encoder import sparse_embed_query
from indexing.vector_store import COLLECTION_NAME


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    title: str
    page_number: int
    section_id: str
    text: str
    score: float


def hybrid_search(
    client: QdrantClient,
    query: str,
    top_k: int = 30,
    doc_type_filter: str | None = None,
) -> list[RetrievedChunk]:
    dense_vec = embed_query(query)
    sparse_vec = sparse_embed_query(query)

    query_filter = None
    if doc_type_filter:
        query_filter = Filter(
            must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type_filter))]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=top_k, filter=query_filter),
            Prefetch(
                query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                using="sparse",
                limit=top_k,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    return [
        RetrievedChunk(
            chunk_id=point.payload["chunk_id"],
            doc_id=point.payload["doc_id"],
            doc_type=point.payload["doc_type"],
            title=point.payload["title"],
            page_number=point.payload["page_number"],
            section_id=point.payload["section_id"],
            text=point.payload["text"],
            score=point.score,
        )
        for point in results.points
    ]
