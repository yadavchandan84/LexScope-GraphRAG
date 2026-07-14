# Legal RAG System — Approved Design

**Date:** 2026-07-11
**Source:** `Legal_RAG_PRD (1).docx` v1.0, adapted after brainstorming with the owner.

## What we're building

A Retrieval-Augmented Generation system over a legal/tax PDF corpus (~500 pages combined across multiple PDFs, growing over time) that answers questions with **verifiable citations** (doc name + page). Hybrid retrieval (dense + BM25) with cross-encoder reranking and citation-graph expansion (GraphRAG), Gemini free-tier generation with key/model rotation, a citation guardrail, a FastAPI backend, a minimal HTML/JS frontend, and a Ragas evaluation harness.

## Decisions made during brainstorming (deviations from the PRD)

| Topic | PRD said | We chose | Why |
|---|---|---|---|
| Vector DB | Local Qdrant install | **Qdrant Cloud free tier** (1GB RAM / 4GB disk cluster, ~1M vector capacity) | Zero local RAM cost; laptop has 16GB with ~7GB free |
| Keyword/BM25 | Local Elasticsearch | **Qdrant sparse vectors (FastEmbed BM25)** in the same collection, native server-side RRF fusion | Drops a whole JVM service; one database does true hybrid search |
| Citation graph | Local Neo4j Community | **Neo4j AuraDB Free** (≥50k nodes / 175k rels — far more than needed) | Zero local install; corpus graph is tiny |
| Orchestration | LangChain LCEL | **Plain Python functions** | Easier to debug, fewer deps; nothing here needs LCEL |
| Corpus | 100 PDFs | **Multiple PDFs, ~500 pages combined**, more added later | Owner's actual corpus; ingestion must be idempotent/re-runnable |
| Frontend | HTML/JS or React | **Plain HTML/CSS/JS**, no build step | "Don't make it too complex" |

Unchanged from the PRD: PyMuPDF parsing, structure-aware chunking, `BAAI/bge-large-en-v1.5` embeddings (local, GTX 1650 4GB VRAM), `BAAI/bge-reranker-large` cross-encoder, RRF fusion, 1-hop graph expansion, Gemini multi-key/model rotation, structured-JSON prompts, citation guardrail, FastAPI `POST /query`, Ragas evaluation.

## Architecture

```
[PDFs in PDFS/]
      |
      v
[INGESTION]  PyMuPDF parse (page-aware) -> classify doc_type -> structure-aware chunk
      |
      +---------------------------+
      v                           v
[Qdrant Cloud]               [Neo4j Aura]
dense (BGE-large) +          (:Chunk)-[:CITES]->(:Ref)-[:DEFINED_IN]->(:Chunk)
sparse (BM25) vectors        citation graph
      \                           /
       v                         v
[QUERY] hybrid search (dense + sparse, server-side RRF) -> top 30
      -> bge-reranker-large -> top 5-8
      -> 1-hop graph expansion (cited sections, max +10 chunks)
      -> Gemini (key/model rotation, JSON schema output)
      -> citation guardrail (verify every citation against sent context)
      -> FastAPI /query -> frontend
```

## Environment facts

- Laptop: 16GB RAM, GTX 1650 Mobile (4GB VRAM), 16 cores, Python 3.10, Linux.
- Only Python + two BGE models run locally; both databases are hosted free tiers.
- Qdrant free clusters **suspend after 1 week idle** (resume from console); Aura Free also pauses when idle. Treat "reconnect after resume" as a normal flow.
- Owner has multiple Gemini free-tier API keys ready.

## Success metrics (from PRD)

- Context Recall ≥ 0.85 on the golden set; Faithfulness ≥ 0.90.
- 100% of returned citations pass the guardrail.
- End-to-end query latency under ~8–10s locally.
- `insufficient_context` correctly surfaces for out-of-corpus questions.

## Out of scope (v1)

Auth, multi-tenancy, billing, model fine-tuning, scheduled ingestion, React, Docker, LangChain.

## Implementation plan

See [plan.md](plan.md) — 7 phases, retrieval-first ordering.
