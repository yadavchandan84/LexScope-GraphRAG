import json

from neo4j import Driver
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from indexing.citation_extractor import extract_citations
from indexing.embedder import embed_documents
from indexing.graph_store import ensure_schema, get_neo4j_driver
from indexing.sparse_encoder import sparse_embed_documents
from indexing.vector_store import COLLECTION_NAME, ensure_collection, get_qdrant_client

CHUNKS_PATH = "ingestion/chunks.json"


def load_chunks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_chunks_to_qdrant(client: QdrantClient, chunks: list[dict]) -> int:
    texts = [c["text"] for c in chunks]
    dense_vecs = embed_documents(texts)
    sparse_vecs = sparse_embed_documents(texts)

    points = []
    for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector={"dense": dense, "sparse": sparse},
                payload=chunk,
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def index_chunks_to_neo4j(driver: Driver, chunks: list[dict]) -> int:
    edge_count = 0
    with driver.session() as session:
        for chunk in chunks:
            session.run(
                "MERGE (c:Chunk {chunk_id: $chunk_id}) "
                "SET c.doc_id = $doc_id, c.page_number = $page_number, c.section_id = $section_id",
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                page_number=chunk["page_number"],
                section_id=chunk["section_id"],
            )

            if chunk["doc_type"] in ("Act", "Tax") and chunk["section_id"].startswith("Section"):
                session.run(
                    "MATCH (c:Chunk {chunk_id: $chunk_id}) "
                    "MERGE (r:Ref {ref_key: $ref_key}) "
                    "MERGE (r)-[:DEFINED_IN]->(c)",
                    chunk_id=chunk["chunk_id"],
                    ref_key=chunk["section_id"],
                )

            citations = extract_citations(chunk["text"])
            for ref in citations:
                session.run(
                    "MATCH (c:Chunk {chunk_id: $chunk_id}) "
                    "MERGE (r:Ref {ref_key: $ref_key}) "
                    "MERGE (c)-[:CITES]->(r)",
                    chunk_id=chunk["chunk_id"],
                    ref_key=ref.ref_key,
                )
                edge_count += 1
    return edge_count


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)

    qdrant_client = get_qdrant_client()
    ensure_collection(qdrant_client)
    qdrant_count = index_chunks_to_qdrant(qdrant_client, chunks)
    print(f"Upserted {qdrant_count} points to Qdrant collection.")

    neo4j_driver = get_neo4j_driver()
    ensure_schema(neo4j_driver)
    edge_count = index_chunks_to_neo4j(neo4j_driver, chunks)
    neo4j_driver.close()
    print(f"Created/merged {edge_count} CITES edges in Neo4j.")
