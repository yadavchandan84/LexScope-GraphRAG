import json
import os
from dataclasses import asdict

from ingestion.chunker import Chunk, chunk_document
from ingestion.classifier import classify_document
from ingestion.parser import parse_pdf

OUTPUT_PATH = os.path.join("ingestion", "chunks.json")


def discover_pdfs(folder: str) -> list[str]:
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".pdf")
    )


def ingest_folder(folder: str) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in discover_pdfs(folder):
        filename = os.path.basename(path)
        doc_id = os.path.splitext(filename)[0]
        pages = parse_pdf(path)
        doc_type = classify_document(pages, filename)
        chunks = chunk_document(pages, doc_type, doc_id=doc_id, title=doc_id)
        all_chunks.extend(chunks)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2)

    return all_chunks


def _print_summary(chunks: list[Chunk]) -> None:
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.doc_type] = counts.get(c.doc_type, 0) + 1
    print(f"Ingested {len(chunks)} chunks total:")
    for doc_type, count in sorted(counts.items()):
        print(f"  {doc_type}: {count}")


if __name__ == "__main__":
    folder = "PDFS" if os.path.isdir("PDFS") else "pdfs"
    chunks = ingest_folder(folder)
    _print_summary(chunks)
