from neo4j import Driver
from qdrant_client import QdrantClient

from retrieval.hybrid_retriever import RetrievedChunk
from indexing.vector_store import COLLECTION_NAME


def expand_with_citations(
    driver: Driver,
    qdrant_client: QdrantClient,
    chunks: list[RetrievedChunk],
    limit: int = 10,
) -> list[RetrievedChunk]:
    if not chunks:
        return chunks

    existing_ids = {c.chunk_id for c in chunks}
    top_ids = [c.chunk_id for c in chunks]

    with driver.session() as session:
        result = session.run(
            "MATCH (c:Chunk)-[:CITES]->(r:Ref)<-[:DEFINED_IN]-(target:Chunk) "
            "WHERE c.chunk_id IN $top_ids AND NOT target.chunk_id IN $existing_ids "
            "RETURN DISTINCT target.chunk_id AS chunk_id "
            "LIMIT $limit",
            top_ids=top_ids,
            existing_ids=list(existing_ids),
            limit=limit,
        )
        new_chunk_ids = [record["chunk_id"] for record in result]

    if not new_chunk_ids:
        return chunks

    points = qdrant_client.retrieve(
        collection_name=COLLECTION_NAME, ids=new_chunk_ids, with_payload=True
    )

    expanded_chunks = [
        RetrievedChunk(
            chunk_id=p.payload["chunk_id"],
            doc_id=p.payload["doc_id"],
            doc_type=p.payload["doc_type"],
            title=p.payload["title"],
            page_number=p.payload["page_number"],
            section_id=p.payload["section_id"],
            text=p.payload["text"],
            score=0.0,
        )
        for p in points
    ]

    return chunks + expanded_chunks
