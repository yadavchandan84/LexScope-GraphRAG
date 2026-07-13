import sys

from indexing.graph_store import get_neo4j_driver
from indexing.vector_store import get_qdrant_client
from retrieval.graph_expander import expand_with_citations
from retrieval.hybrid_retriever import RetrievedChunk, hybrid_search
from retrieval.reranker import rerank


def run_retrieval_pipeline(
    query: str, doc_type_filter: str | None = None, rerank_top_k: int = 8
) -> list[RetrievedChunk]:
    qdrant_client = get_qdrant_client()
    candidates = hybrid_search(qdrant_client, query, top_k=30, doc_type_filter=doc_type_filter)
    top = rerank(query, candidates, top_k=rerank_top_k)

    driver = get_neo4j_driver()
    try:
        expanded = expand_with_citations(driver, qdrant_client, top, limit=10)
    finally:
        driver.close()

    return expanded


def _print_results(results: list[RetrievedChunk]) -> None:
    for i, chunk in enumerate(results, start=1):
        snippet = chunk.text[:120].replace("\n", " ")
        print(f"[{i}] {chunk.doc_id} p.{chunk.page_number} ({chunk.section_id}) score={chunk.score:.4f}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    query_text = " ".join(sys.argv[1:])
    if not query_text:
        print("Usage: python -m retrieval.cli \"your query\"")
        sys.exit(1)
    results = run_retrieval_pipeline(query_text)
    _print_results(results)
