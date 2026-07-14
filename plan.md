# Legal RAG System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-laptop, $0-cost legal/tax RAG system that retrieves with hybrid (dense+BM25) search, reranks, expands via a citation graph, generates grounded answers with Gemini, verifies citations, and serves it all through a FastAPI + plain-HTML frontend, measured with Ragas.

**Architecture:** PDFs → PyMuPDF parse (page-aware) → doc-type classify → structure-aware chunk → dual-index (Qdrant Cloud dense+sparse vectors, Neo4j Aura citation graph) → query time: hybrid search+RRF → cross-encoder rerank → 1-hop graph expansion → Gemini generation (key/model rotation) → citation guardrail → FastAPI `/query` → HTML/JS frontend. See [design.md](design.md) for full rationale and deviations from the original PRD.

**Tech Stack:** Python 3.10, PyMuPDF (fitz), `qdrant-client` (Qdrant Cloud, dense+sparse hybrid), `neo4j` driver (AuraDB Free), `sentence-transformers` (BGE-large embeddings + BGE-reranker-large, local CPU/GPU), `google-generativeai` (Gemini), FastAPI + Uvicorn, plain HTML/CSS/JS, `ragas` + `datasets` for evaluation, `pytest` for tests.

## Global Constraints

- Python 3.10 (matches installed interpreter — do not require 3.11+ features).
- Zero paid services: Qdrant Cloud free tier, Neo4j AuraDB free tier, Gemini free tier only.
- No Docker, no LangChain/LCEL, no React — plain Python modules and vanilla JS.
- No local database servers — only two locally-run ML models (embedder, reranker).
- All chunk metadata must include: `chunk_id, doc_id, doc_type, title, page_number, section_id, text`.
- Ingestion must be idempotent — re-running on a corpus with new PDFs added must not duplicate existing chunks (upsert keyed on a stable content hash, not random UUID).
- BGE embeddings: raw chunk text at index time; query text prefixed with `"Represent this question for retrieval: "` at query time.
- Every citation returned to the user must be verified against the exact context sent to Gemini for that request before the API responds (citation guardrail — no silent pass-through).
- Gemini generation temperature 0–0.2.
- Secrets (Gemini keys, Qdrant/Neo4j credentials) live only in `.env`, never committed; `.gitignore` must include `.env`.

---

## Phase 1: Project Setup & Cloud Connectivity

**Goal of phase:** A working Python environment with verified connections to Qdrant Cloud and Neo4j Aura, and a config module every later phase imports from.

### Task 1.1: Repository scaffold and dependency setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `config.py` exposes `Settings` (a `pydantic.BaseSettings` or plain dataclass) with fields: `qdrant_url: str`, `qdrant_api_key: str`, `neo4j_uri: str`, `neo4j_user: str`, `neo4j_password: str`, `gemini_api_keys: list[str]`, `gemini_models: list[str]`, and a module-level `settings = Settings()` singleton loaded from environment variables (via `python-dotenv`).

- [ ] **Step 1: Create `requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.2
pydantic-settings==2.5.2
python-dotenv==1.0.1
pymupdf==1.24.11
qdrant-client[fastembed]==1.11.3
neo4j==5.25.0
sentence-transformers==3.2.1
torch==2.4.1
google-generativeai==0.8.3
ragas==0.2.6
datasets==3.0.1
pytest==8.3.3
httpx==0.27.2
```

- [ ] **Step 2: Create `.env.example`**

```
QDRANT_URL=https://xxxxx.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-aura-password
GEMINI_API_KEYS=key1,key2,key3
GEMINI_MODELS=gemini-2.5-flash,gemini-2.0-flash,gemini-2.5-flash-lite
```

- [ ] **Step 3: Create `.gitignore`**

```
.env
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
.pytest_cache/
evaluation/report.csv
```

- [ ] **Step 4: Create `config.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.qdrant_url = self._require("QDRANT_URL")
        self.qdrant_api_key = self._require("QDRANT_API_KEY")
        self.neo4j_uri = self._require("NEO4J_URI")
        self.neo4j_user = self._require("NEO4J_USER")
        self.neo4j_password = self._require("NEO4J_PASSWORD")
        self.gemini_api_keys = [
            k.strip() for k in self._require("GEMINI_API_KEYS").split(",") if k.strip()
        ]
        self.gemini_models = [
            m.strip() for m in self._require("GEMINI_MODELS").split(",") if m.strip()
        ]

    @staticmethod
    def _require(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value


settings = Settings()
```

- [ ] **Step 5: Write the failing test**

```python
# tests/test_config.py
import os
import pytest


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://test.cloud.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://test.databases.neo4j.io")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-pass")
    monkeypatch.setenv("GEMINI_API_KEYS", "k1,k2")
    monkeypatch.setenv("GEMINI_MODELS", "gemini-2.5-flash,gemini-2.0-flash")

    import importlib
    import config
    importlib.reload(config)

    assert config.settings.qdrant_url == "https://test.cloud.qdrant.io"
    assert config.settings.gemini_api_keys == ["k1", "k2"]
    assert config.settings.gemini_models == ["gemini-2.5-flash", "gemini-2.0-flash"]


def test_settings_raises_on_missing_var(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    import importlib
    import config
    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        importlib.reload(config)
```

- [ ] **Step 6: Set up virtualenv and install dependencies**

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Expected: all packages install without error (torch CPU or CUDA wheel depending on platform detection).

- [ ] **Step 7: Copy `.env.example` to `.env` and fill in real credentials**

Run: `cp .env.example .env`
Then manually edit `.env` with real Qdrant Cloud URL/key, Neo4j Aura URI/user/password, and Gemini API keys (comma-separated, no spaces needed).
Expected: `.env` exists and is git-ignored (verify with `git check-ignore .env` once repo is initialized in Task 1.4).

- [ ] **Step 8: Run the config tests**

Run: `pytest tests/test_config.py -v`
Expected: both tests PASS (the monkeypatch tests don't depend on the real `.env`).

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .env.example .gitignore config.py tests/test_config.py
git commit -m "feat: project scaffold and settings loader"
```

---

### Task 1.2: Qdrant Cloud connectivity + collection bootstrap

**Files:**
- Create: `indexing/__init__.py`
- Create: `indexing/vector_store.py`
- Create: `tests/test_vector_store.py`

**Interfaces:**
- Consumes: `config.settings` (Task 1.1) for `qdrant_url`, `qdrant_api_key`.
- Produces: `indexing/vector_store.py` exposes:
  - `get_qdrant_client() -> QdrantClient`
  - `COLLECTION_NAME = "legal_chunks"`
  - `ensure_collection(client: QdrantClient) -> None` — creates the collection with a named dense vector `"dense"` (size 1024, distance COSINE, matching `bge-large-en-v1.5`) and a named sparse vector `"sparse"`, if it doesn't already exist.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vector_store.py
from indexing.vector_store import get_qdrant_client, ensure_collection, COLLECTION_NAME


def test_ensure_collection_creates_and_is_idempotent():
    client = get_qdrant_client()
    ensure_collection(client)
    assert client.collection_exists(COLLECTION_NAME)

    # calling again must not raise or duplicate
    ensure_collection(client)
    info = client.get_collection(COLLECTION_NAME)
    assert "dense" in info.config.params.vectors
    assert "sparse" in info.config.params.sparse_vectors
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vector_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'indexing'`

- [ ] **Step 3: Create `indexing/__init__.py`** (empty file)

- [ ] **Step 4: Write `indexing/vector_store.py`**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
)

from config import settings

COLLECTION_NAME = "legal_chunks"
DENSE_VECTOR_SIZE = 1024  # bge-large-en-v1.5 output dimension


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_vector_store.py -v`
Expected: PASS (requires real `.env` credentials to be valid — this test hits live Qdrant Cloud)

- [ ] **Step 6: Commit**

```bash
git add indexing/__init__.py indexing/vector_store.py tests/test_vector_store.py
git commit -m "feat: Qdrant Cloud client and hybrid collection bootstrap"
```

---

### Task 1.3: Neo4j Aura connectivity + schema constraints

**Files:**
- Create: `indexing/graph_store.py`
- Create: `tests/test_graph_store.py`

**Interfaces:**
- Consumes: `config.settings` for `neo4j_uri`, `neo4j_user`, `neo4j_password`.
- Produces: `indexing/graph_store.py` exposes:
  - `get_neo4j_driver() -> neo4j.Driver`
  - `ensure_schema(driver) -> None` — creates uniqueness constraints on `Chunk.chunk_id` and `Ref.ref_key`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_store.py
from indexing.graph_store import get_neo4j_driver, ensure_schema


def test_ensure_schema_and_connectivity():
    driver = get_neo4j_driver()
    driver.verify_connectivity()
    ensure_schema(driver)

    with driver.session() as session:
        result = session.run("SHOW CONSTRAINTS YIELD name RETURN collect(name) AS names")
        names = result.single()["names"]
        assert "chunk_id_unique" in names
        assert "ref_key_unique" in names
    driver.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'indexing.graph_store'`

- [ ] **Step 3: Write `indexing/graph_store.py`**

```python
from neo4j import Driver, GraphDatabase

from config import settings


def get_neo4j_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def ensure_schema(driver: Driver) -> None:
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT ref_key_unique IF NOT EXISTS "
            "FOR (r:Ref) REQUIRE r.ref_key IS UNIQUE"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_store.py -v`
Expected: PASS (hits live Neo4j Aura instance)

- [ ] **Step 5: Commit**

```bash
git add indexing/graph_store.py tests/test_graph_store.py
git commit -m "feat: Neo4j Aura client and schema constraints"
```

---

### Task 1.4: Git init

**Files:** none (repo-level operation)

- [ ] **Step 1: Initialize git repository if not already**

Run: `git init` (skip if already a repo)

- [ ] **Step 2: Verify `.env` is ignored**

Run: `git check-ignore .env`
Expected: prints `.env` (confirms it's ignored)

- [ ] **Step 3: Stage and commit everything from Phase 1**

Run:
```bash
git add -A
git commit -m "chore: phase 1 complete — scaffold, Qdrant + Neo4j connectivity"
```

---

## Phase 2: Ingestion (Parsing + Chunking)

**Goal of phase:** Turn PDFs in `PDFS/` into a validated list of metadata-rich chunks, saved to disk as JSON, ready for indexing. No database writes yet — this phase is fully offline/testable.

### Task 2.1: PDF parser with page-aware text extraction

**Files:**
- Create: `ingestion/__init__.py`
- Create: `ingestion/parser.py`
- Create: `tests/test_parser.py`
- Create: `tests/fixtures/sample.pdf` (a tiny 2-page PDF generated by the test setup, not hand-authored)

**Interfaces:**
- Produces: `ingestion/parser.py` exposes:
  - `PageText` — a dataclass with fields `page_number: int, text: str`
  - `parse_pdf(path: str) -> list[PageText]` — one entry per page, 1-indexed page numbers, in order.

- [ ] **Step 1: Write the failing test (with a generated fixture)**

```python
# tests/test_parser.py
import fitz  # PyMuPDF
import pytest

from ingestion.parser import parse_pdf, PageText


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Section 1. This is page one content.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Section 2. This is page two content.")
    doc.save(str(path))
    doc.close()
    return str(path)


def test_parse_pdf_returns_one_entry_per_page(sample_pdf):
    pages = parse_pdf(sample_pdf)
    assert len(pages) == 2
    assert all(isinstance(p, PageText) for p in pages)
    assert pages[0].page_number == 1
    assert "page one" in pages[0].text
    assert pages[1].page_number == 2
    assert "page two" in pages[1].text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion'`

- [ ] **Step 3: Create `ingestion/__init__.py`** (empty file)

- [ ] **Step 4: Write `ingestion/parser.py`**

```python
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class PageText:
    page_number: int
    text: str


def parse_pdf(path: str) -> list[PageText]:
    doc = fitz.open(path)
    try:
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(PageText(page_number=i + 1, text=text))
        return pages
    finally:
        doc.close()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_parser.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ingestion/__init__.py ingestion/parser.py tests/test_parser.py
git commit -m "feat: page-aware PDF parsing with PyMuPDF"
```

---

### Task 2.2: Document type classifier

**Files:**
- Create: `ingestion/classifier.py`
- Create: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `list[PageText]` from Task 2.1's `parse_pdf`.
- Produces: `ingestion/classifier.py` exposes:
  - `DocType` — a `str` Enum with values `ACT = "Act"`, `JUDGMENT = "Judgment"`, `POV = "POV"`, `TAX = "Tax"`.
  - `classify_document(pages: list[PageText], filename: str) -> DocType` — rule-based keyword classifier.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier.py
from ingestion.classifier import classify_document, DocType
from ingestion.parser import PageText


def test_classifies_tax_document_by_keywords():
    pages = [PageText(1, "This publication explains the assessee's deduction under Section 80C.")]
    assert classify_document(pages, "irs_pub_502.pdf") == DocType.TAX


def test_classifies_judgment_by_keywords():
    pages = [PageText(1, "J U D G M E N T\n\n1. The appellant filed this appeal against the order.")]
    assert classify_document(pages, "case_2019.pdf") == DocType.JUDGMENT


def test_classifies_act_by_keywords():
    pages = [PageText(1, "WHEREAS it is expedient to consolidate the law, Section 1 of this Act...")]
    assert classify_document(pages, "income_tax_act.pdf") == DocType.ACT


def test_defaults_to_pov_when_no_keywords_match():
    pages = [PageText(1, "In my opinion, the market trends suggest a shift in consumer behavior.")]
    assert classify_document(pages, "opinion_piece.pdf") == DocType.POV
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.classifier'`

- [ ] **Step 3: Write `ingestion/classifier.py`**

```python
import re
from enum import Enum

from ingestion.parser import PageText


class DocType(str, Enum):
    ACT = "Act"
    JUDGMENT = "Judgment"
    POV = "POV"
    TAX = "Tax"


_JUDGMENT_PATTERNS = [r"J\s*U\s*D\s*G\s*M\s*E\s*N\s*T", r"\bappellant\b", r"\brespondent\b"]
_ACT_PATTERNS = [r"\bWHEREAS\b", r"\bthis Act\b", r"\bSection\s+\d+\b.*\bAct\b"]
_TAX_PATTERNS = [r"\bassessee\b", r"\bSection\s+80[A-Z]?\b", r"\btax(able|payer)?\b", r"\bIRS\b", r"\bdeduction\b"]


def classify_document(pages: list[PageText], filename: str) -> DocType:
    sample_text = " ".join(p.text for p in pages[:3])

    if _matches_any(sample_text, _JUDGMENT_PATTERNS) or "judgment" in filename.lower():
        return DocType.JUDGMENT
    if _matches_any(sample_text, _TAX_PATTERNS) or "tax" in filename.lower() or "irs" in filename.lower():
        return DocType.TAX
    if _matches_any(sample_text, _ACT_PATTERNS) or "act" in filename.lower():
        return DocType.ACT
    return DocType.POV


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingestion/classifier.py tests/test_classifier.py
git commit -m "feat: rule-based document type classifier"
```

---

### Task 2.3: Structure-aware chunker

**Files:**
- Create: `ingestion/chunker.py`
- Create: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `list[PageText]` (Task 2.1), `DocType` (Task 2.2).
- Produces: `ingestion/chunker.py` exposes:
  - `Chunk` — dataclass: `chunk_id: str, doc_id: str, doc_type: str, title: str, page_number: int, section_id: str, text: str`
  - `chunk_document(pages: list[PageText], doc_type: DocType, doc_id: str, title: str) -> list[Chunk]` — dispatches to section-based chunking for `ACT`/`TAX`, numbered-paragraph chunking for `JUDGMENT`, and paragraph chunking (300–500 tokens, ~15% overlap) for `POV`.
  - `make_chunk_id(doc_id: str, section_id: str, text: str) -> str` — a stable SHA-256-based id (content hash) so re-ingestion is idempotent (see Global Constraints).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
from ingestion.chunker import chunk_document, make_chunk_id
from ingestion.classifier import DocType
from ingestion.parser import PageText


def test_act_chunking_splits_on_sections_and_keeps_page_number():
    pages = [
        PageText(1, "Section 1. Short title.\nThis Act may be called the Test Act.\n"
                    "Section 2. Definitions.\nIn this Act, unless the context otherwise requires...")
    ]
    chunks = chunk_document(pages, DocType.ACT, doc_id="test-act", title="Test Act")
    assert len(chunks) == 2
    assert chunks[0].section_id == "Section 1"
    assert "Short title" in chunks[0].text
    assert chunks[1].section_id == "Section 2"
    assert all(c.page_number == 1 for c in chunks)
    assert all(c.doc_type == "Act" for c in chunks)


def test_judgment_chunking_splits_on_numbered_paragraphs():
    pages = [
        PageText(1, "1. The appellant challenges the order dated 5 May.\n"
                    "2. The respondent contends the order is valid.\n")
    ]
    chunks = chunk_document(pages, DocType.JUDGMENT, doc_id="test-case", title="Test v. Case")
    assert len(chunks) == 2
    assert chunks[0].section_id == "para_1"
    assert chunks[1].section_id == "para_2"


def test_chunk_id_is_stable_for_same_content():
    id1 = make_chunk_id("doc1", "Section 1", "some text")
    id2 = make_chunk_id("doc1", "Section 1", "some text")
    id3 = make_chunk_id("doc1", "Section 1", "different text")
    assert id1 == id2
    assert id1 != id3


def test_pov_chunking_produces_token_bounded_chunks_with_overlap():
    long_para = " ".join(["word"] * 900)  # ~900 tokens of filler
    pages = [PageText(1, long_para)]
    chunks = chunk_document(pages, DocType.POV, doc_id="test-pov", title="Test Opinion")
    assert len(chunks) >= 2
    for c in chunks:
        word_count = len(c.text.split())
        assert 200 <= word_count <= 600  # ~300-500 target with overlap slack
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.chunker'`

- [ ] **Step 3: Write `ingestion/chunker.py`**

```python
import hashlib
import re
from dataclasses import dataclass

from ingestion.classifier import DocType
from ingestion.parser import PageText

_SECTION_RE = re.compile(r"(?=^(?:Section|§|Rule)\s+\d+[A-Za-z]?\.?)", re.MULTILINE)
_SECTION_LABEL_RE = re.compile(r"^(Section|§|Rule)\s+(\d+[A-Za-z]?)", re.MULTILINE)
_PARA_RE = re.compile(r"(?=^\d+\.\s)", re.MULTILINE)
_PARA_LABEL_RE = re.compile(r"^(\d+)\.\s")

_POV_TARGET_WORDS = 400
_POV_OVERLAP_WORDS = 60


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    title: str
    page_number: int
    section_id: str
    text: str


def make_chunk_id(doc_id: str, section_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{doc_id}|{section_id}|{text}".encode("utf-8")).hexdigest()
    return digest[:24]


def chunk_document(pages: list[PageText], doc_type: DocType, doc_id: str, title: str) -> list[Chunk]:
    if doc_type in (DocType.ACT, DocType.TAX):
        return _chunk_by_section(pages, doc_type, doc_id, title)
    if doc_type == DocType.JUDGMENT:
        return _chunk_by_paragraph_number(pages, doc_type, doc_id, title)
    return _chunk_pov(pages, doc_type, doc_id, title)


def _chunk_by_section(pages: list[PageText], doc_type: DocType, doc_id: str, title: str) -> list[Chunk]:
    chunks = []
    for page in pages:
        parts = [p for p in _SECTION_RE.split(page.text) if p.strip()]
        if not parts:
            continue
        for part in parts:
            label_match = _SECTION_LABEL_RE.match(part.strip())
            section_id = f"{label_match.group(1)} {label_match.group(2)}" if label_match else "preamble"
            text = part.strip()
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, section_id, text),
                    doc_id=doc_id,
                    doc_type=doc_type.value,
                    title=title,
                    page_number=page.page_number,
                    section_id=section_id,
                    text=text,
                )
            )
    return chunks


def _chunk_by_paragraph_number(pages: list[PageText], doc_type: DocType, doc_id: str, title: str) -> list[Chunk]:
    chunks = []
    for page in pages:
        parts = [p for p in _PARA_RE.split(page.text) if p.strip()]
        if not parts:
            continue
        for part in parts:
            label_match = _PARA_LABEL_RE.match(part.strip())
            section_id = f"para_{label_match.group(1)}" if label_match else "unnumbered"
            text = part.strip()
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, section_id, text),
                    doc_id=doc_id,
                    doc_type=doc_type.value,
                    title=title,
                    page_number=page.page_number,
                    section_id=section_id,
                    text=text,
                )
            )
    return chunks


def _chunk_pov(pages: list[PageText], doc_type: DocType, doc_id: str, title: str) -> list[Chunk]:
    chunks = []
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        start = 0
        idx = 0
        step = _POV_TARGET_WORDS - _POV_OVERLAP_WORDS
        while start < len(words):
            window = words[start:start + _POV_TARGET_WORDS]
            text = " ".join(window)
            section_id = f"para_block_{idx}"
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, section_id, text),
                    doc_id=doc_id,
                    doc_type=doc_type.value,
                    title=title,
                    page_number=page.page_number,
                    section_id=section_id,
                    text=text,
                )
            )
            start += step
            idx += 1
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: structure-aware chunker for Act/Tax, Judgment, and POV documents"
```

---

### Task 2.4: Ingestion orchestrator over `PDFS/`

**Files:**
- Create: `ingestion/run_ingest.py`
- Create: `tests/test_run_ingest.py`

**Interfaces:**
- Consumes: `parse_pdf` (2.1), `classify_document`/`DocType` (2.2), `chunk_document`/`Chunk` (2.3).
- Produces: `ingestion/run_ingest.py` exposes:
  - `discover_pdfs(folder: str) -> list[str]` — sorted list of `.pdf` paths.
  - `ingest_folder(folder: str) -> list[Chunk]` — parses+classifies+chunks every PDF, returns the combined chunk list, and writes it to `ingestion/chunks.json` (list of dicts).
  - CLI entry: `python -m ingestion.run_ingest` runs `ingest_folder("PDFS")` and prints a per-doc-type chunk count summary.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_ingest.py
import json

import fitz
import pytest

from ingestion.run_ingest import discover_pdfs, ingest_folder


@pytest.fixture
def pdf_folder(tmp_path):
    folder = tmp_path / "pdfs"
    folder.mkdir()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Section 1. This publication covers assessee deductions under tax law.")
    doc.save(str(folder / "irs_pub_1.pdf"))
    doc.close()

    doc2 = fitz.open()
    page2 = doc2.new_page()
    page2.insert_text((72, 72), "1. The appellant raises this appeal against the respondent's order.")
    doc2.save(str(folder / "case_1.pdf"))
    doc2.close()
    return str(folder)


def test_discover_pdfs_finds_all_pdf_files(pdf_folder):
    files = discover_pdfs(pdf_folder)
    assert len(files) == 2
    assert all(f.endswith(".pdf") for f in files)


def test_ingest_folder_returns_chunks_and_writes_json(pdf_folder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ingestion").mkdir()
    chunks = ingest_folder(pdf_folder)
    assert len(chunks) >= 2

    output_path = tmp_path / "ingestion" / "chunks.json"
    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
    assert len(data) == len(chunks)
    assert "chunk_id" in data[0]
    assert "doc_type" in data[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.run_ingest'`

- [ ] **Step 3: Write `ingestion/run_ingest.py`**

```python
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
    with open(OUTPUT_PATH, "w") as f:
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
    chunks = ingest_folder("PDFS")
    _print_summary(chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Run against the real corpus and verify chunk counts/metadata by hand**

Run: `python -m ingestion.run_ingest`
Expected: prints a per-doc-type summary; inspect `ingestion/chunks.json` and confirm every chunk has non-empty `text`, a plausible `page_number`, and a `doc_type` that matches the source PDF (the 4 IRS publications should classify as `Tax`).

- [ ] **Step 6: Commit**

```bash
git add ingestion/run_ingest.py tests/test_run_ingest.py
git commit -m "feat: ingestion orchestrator producing validated chunks.json"
```

---

## Phase 3: Indexing

**Goal of phase:** Every chunk from `ingestion/chunks.json` is embedded and upserted into Qdrant (dense+sparse), and citation edges are extracted and written to Neo4j. Idempotent — re-running does not duplicate.

### Task 3.1: Embedder wrapper (BGE-large)

**Files:**
- Create: `indexing/embedder.py`
- Create: `tests/test_embedder.py`

**Interfaces:**
- Produces: `indexing/embedder.py` exposes:
  - `get_embedder() -> SentenceTransformer` — loads `BAAI/bge-large-en-v1.5` once (module-level cache).
  - `embed_documents(texts: list[str]) -> list[list[float]]` — raw text, no prefix.
  - `embed_query(text: str) -> list[float]` — prefixes with `"Represent this question for retrieval: "` before embedding.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedder.py
from indexing.embedder import embed_documents, embed_query


def test_embed_documents_returns_1024_dim_vectors():
    vectors = embed_documents(["Section 1. Short title.", "Section 2. Definitions."])
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024


def test_embed_query_uses_instruction_prefix_and_differs_from_raw():
    q = "What does Section 80C say?"
    vec_with_prefix = embed_query(q)
    vec_raw = embed_documents([q])[0]
    assert len(vec_with_prefix) == 1024
    assert vec_with_prefix != vec_raw
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'indexing.embedder'`

- [ ] **Step 3: Write `indexing/embedder.py`**

```python
from functools import lru_cache

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_QUERY_PREFIX = "Represent this question for retrieval: "


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    model = get_embedder()
    return model.encode(_QUERY_PREFIX + text, normalize_embeddings=True).tolist()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedder.py -v`
Expected: PASS (first run downloads the model, ~1.3GB; subsequent runs use the local cache)

- [ ] **Step 5: Commit**

```bash
git add indexing/embedder.py tests/test_embedder.py
git commit -m "feat: BGE-large embedder with asymmetric query prefixing"
```

---

### Task 3.2: Sparse (BM25) vector generation via FastEmbed

**Files:**
- Create: `indexing/sparse_encoder.py`
- Create: `tests/test_sparse_encoder.py`

**Interfaces:**
- Produces: `indexing/sparse_encoder.py` exposes:
  - `get_sparse_model() -> SparseTextEmbedding` — loads `Qdrant/bm25` FastEmbed sparse model (module-level cache).
  - `sparse_embed_documents(texts: list[str]) -> list[SparseVector]` — returns Qdrant `SparseVector` objects (`indices`, `values`).
  - `sparse_embed_query(text: str) -> SparseVector`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sparse_encoder.py
from indexing.sparse_encoder import sparse_embed_documents, sparse_embed_query


def test_sparse_embed_documents_returns_sparse_vectors_with_indices_and_values():
    vecs = sparse_embed_documents(["Section 80C allows a deduction.", "Article 21 protects life and liberty."])
    assert len(vecs) == 2
    assert len(vecs[0].indices) > 0
    assert len(vecs[0].indices) == len(vecs[0].values)


def test_sparse_embed_query_returns_sparse_vector():
    vec = sparse_embed_query("What does Section 80C say?")
    assert len(vec.indices) > 0
    assert len(vec.indices) == len(vec.values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sparse_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'indexing.sparse_encoder'`

- [ ] **Step 3: Write `indexing/sparse_encoder.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sparse_encoder.py -v`
Expected: PASS (FastEmbed downloads a small BM25 model on first run)

- [ ] **Step 5: Commit**

```bash
git add indexing/sparse_encoder.py tests/test_sparse_encoder.py
git commit -m "feat: BM25 sparse vector encoding via FastEmbed"
```

---

### Task 3.3: Citation extractor

**Files:**
- Create: `indexing/citation_extractor.py`
- Create: `tests/test_citation_extractor.py`

**Interfaces:**
- Produces: `indexing/citation_extractor.py` exposes:
  - `CitationRef` — dataclass: `ref_type: str` (`"statute"` or `"case"`), `ref_key: str` (normalized, e.g. `"Section 80C"` or `"Smith v. Jones"`), `raw_text: str`.
  - `extract_citations(text: str) -> list[CitationRef]` — regex-based extraction of statute refs (`§N`, `Section N`, `Article N`) and case refs (`X v. Y`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_citation_extractor.py
from indexing.citation_extractor import extract_citations


def test_extracts_statute_section_reference():
    refs = extract_citations("As held under Section 80C of the Income Tax Act, a deduction applies.")
    statute_refs = [r for r in refs if r.ref_type == "statute"]
    assert any(r.ref_key == "Section 80C" for r in statute_refs)


def test_extracts_section_symbol_reference():
    refs = extract_citations("The court relied on §162 in reaching its conclusion.")
    assert any(r.ref_key == "Section 162" for r in refs if r.ref_type == "statute")


def test_extracts_case_citation():
    refs = extract_citations("As established in Smith v. Jones, the doctrine applies.")
    case_refs = [r for r in refs if r.ref_type == "case"]
    assert any(r.ref_key == "Smith v. Jones" for r in case_refs)


def test_no_false_positives_on_plain_text():
    refs = extract_citations("This is a plain sentence with no citations at all.")
    assert refs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_citation_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'indexing.citation_extractor'`

- [ ] **Step 3: Write `indexing/citation_extractor.py`**

```python
import re
from dataclasses import dataclass

_STATUTE_PATTERNS = [
    re.compile(r"\bSection\s+(\d+[A-Za-z]?)\b"),
    re.compile(r"§\s?(\d+[A-Za-z]?)\b"),
    re.compile(r"\bArticle\s+(\d+[A-Za-z]?)\b"),
]
_CASE_PATTERN = re.compile(r"\b([A-Z][a-zA-Z.]+(?:\s[A-Z][a-zA-Z.]+)*)\sv\.\s([A-Z][a-zA-Z.]+(?:\s[A-Z][a-zA-Z.]+)*)")


@dataclass
class CitationRef:
    ref_type: str
    ref_key: str
    raw_text: str


def extract_citations(text: str) -> list[CitationRef]:
    refs: list[CitationRef] = []
    seen: set[str] = set()

    for pattern in _STATUTE_PATTERNS:
        for match in pattern.finditer(text):
            number = match.group(1)
            ref_key = f"Section {number}"
            if ref_key not in seen:
                seen.add(ref_key)
                refs.append(CitationRef(ref_type="statute", ref_key=ref_key, raw_text=match.group(0)))

    for match in _CASE_PATTERN.finditer(text):
        ref_key = f"{match.group(1)} v. {match.group(2)}"
        if ref_key not in seen:
            seen.add(ref_key)
            refs.append(CitationRef(ref_type="case", ref_key=ref_key, raw_text=match.group(0)))

    return refs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_citation_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add indexing/citation_extractor.py tests/test_citation_extractor.py
git commit -m "feat: regex-based citation extractor for statutes and case names"
```

---

### Task 3.4: Indexing orchestrator (Qdrant upsert + Neo4j graph writes)

**Files:**
- Create: `indexing/run_index.py`
- Create: `tests/test_run_index.py`

**Interfaces:**
- Consumes: `ingestion/chunks.json` (Task 2.4), `embed_documents` (3.1), `sparse_embed_documents` (3.2), `extract_citations`/`CitationRef` (3.3), `get_qdrant_client`/`ensure_collection`/`COLLECTION_NAME` (1.2), `get_neo4j_driver`/`ensure_schema` (1.3).
- Produces: `indexing/run_index.py` exposes:
  - `load_chunks(path: str) -> list[dict]`
  - `index_chunks_to_qdrant(client, chunks: list[dict]) -> int` — returns number upserted. Point ID = `chunk_id` (deterministic), so re-running upserts in place rather than duplicating.
  - `index_chunks_to_neo4j(driver, chunks: list[dict]) -> int` — returns number of `CITES` edges created. For each chunk, `MERGE (c:Chunk {chunk_id: ...})` then for each extracted citation `MERGE (r:Ref {ref_key: ...}) MERGE (c)-[:CITES]->(r)`, and separately `MERGE` a `(:Ref {ref_key: section_id})-[:DEFINED_IN]->(c)` link when the chunk's own `section_id` looks like a statute section — this is what lets graph expansion later resolve a citation back to the chunk that defines it.
  - CLI entry: `python -m indexing.run_index` runs both and prints counts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_index.py
import json

from indexing.run_index import load_chunks, index_chunks_to_qdrant, index_chunks_to_neo4j
from indexing.vector_store import get_qdrant_client, ensure_collection, COLLECTION_NAME
from indexing.graph_store import get_neo4j_driver, ensure_schema


def _sample_chunks():
    return [
        {
            "chunk_id": "test-chunk-section-80c",
            "doc_id": "test-doc",
            "doc_type": "Tax",
            "title": "Test Tax Doc",
            "page_number": 1,
            "section_id": "Section 80C",
            "text": "Section 80C allows a deduction for specified investments.",
        },
        {
            "chunk_id": "test-chunk-judgment-1",
            "doc_id": "test-case",
            "doc_type": "Judgment",
            "title": "Test v. Case",
            "page_number": 1,
            "section_id": "para_1",
            "text": "The appellant relies on Section 80C to claim the deduction.",
        },
    ]


def test_load_chunks_reads_json(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(_sample_chunks()))
    chunks = load_chunks(str(path))
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "test-chunk-section-80c"


def test_index_chunks_to_qdrant_upserts_and_is_idempotent():
    client = get_qdrant_client()
    ensure_collection(client)
    chunks = _sample_chunks()

    count1 = index_chunks_to_qdrant(client, chunks)
    count2 = index_chunks_to_qdrant(client, chunks)  # re-run, same chunk_ids
    assert count1 == 2
    assert count2 == 2

    collection_info = client.get_collection(COLLECTION_NAME)
    assert collection_info.points_count is not None


def test_index_chunks_to_neo4j_creates_cites_edges():
    driver = get_neo4j_driver()
    ensure_schema(driver)
    chunks = _sample_chunks()

    edge_count = index_chunks_to_neo4j(driver, chunks)
    assert edge_count >= 1

    with driver.session() as session:
        result = session.run(
            "MATCH (c:Chunk {chunk_id: 'test-chunk-judgment-1'})-[:CITES]->(r:Ref {ref_key: 'Section 80C'}) "
            "RETURN count(*) AS cnt"
        )
        assert result.single()["cnt"] == 1
    driver.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'indexing.run_index'`

- [ ] **Step 3: Write `indexing/run_index.py`**

```python
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
_STATUTE_SECTION_ID_RE = None  # kept simple: section_id already matches "Section N" for Act/Tax chunks


def load_chunks(path: str) -> list[dict]:
    with open(path) as f:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_index.py -v`
Expected: PASS (hits live Qdrant + Neo4j)

- [ ] **Step 5: Run against the real corpus**

Run: `python -m indexing.run_index`
Expected: prints upsert count matching total chunks from `ingestion/chunks.json`, and a CITES edge count > 0. Spot-check by querying Qdrant's collection point count in the Qdrant Cloud console and running one Cypher query in the Aura console (`MATCH (c:Chunk)-[:CITES]->(r) RETURN c.chunk_id, r.ref_key LIMIT 10`).

- [ ] **Step 6: Commit**

```bash
git add indexing/run_index.py tests/test_run_index.py
git commit -m "feat: indexing orchestrator — Qdrant upsert + Neo4j citation graph writes"
```

---

## Phase 4: Retrieval Core (Hybrid + Rerank + Graph Expansion)

**Goal of phase:** A CLI-testable retrieval pipeline: hybrid search → RRF → rerank → graph expansion, with no API/UI dependency yet.

### Task 4.1: Hybrid retriever (dense + sparse + server-side RRF)

**Files:**
- Create: `retrieval/__init__.py`
- Create: `retrieval/hybrid_retriever.py`
- Create: `tests/test_hybrid_retriever.py`

**Interfaces:**
- Consumes: `get_qdrant_client`/`COLLECTION_NAME` (1.2), `embed_query` (3.1), `sparse_embed_query` (3.2).
- Produces: `retrieval/hybrid_retriever.py` exposes:
  - `RetrievedChunk` — dataclass: `chunk_id, doc_id, doc_type, title, page_number, section_id, text, score: float`
  - `hybrid_search(client, query: str, top_k: int = 30, doc_type_filter: str | None = None) -> list[RetrievedChunk]` — uses Qdrant's `query_points` with `prefetch` (dense + sparse) and `fusion=Fusion.RRF`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hybrid_retriever.py
from indexing.run_index import index_chunks_to_qdrant
from indexing.vector_store import get_qdrant_client, ensure_collection
from retrieval.hybrid_retriever import hybrid_search, RetrievedChunk


def _seed_chunks():
    return [
        {
            "chunk_id": "hr-test-1",
            "doc_id": "hr-doc",
            "doc_type": "Tax",
            "title": "HR Test Doc",
            "page_number": 1,
            "section_id": "Section 80C",
            "text": "Section 80C allows a deduction for life insurance premiums and PPF contributions.",
        },
        {
            "chunk_id": "hr-test-2",
            "doc_id": "hr-doc",
            "doc_type": "Tax",
            "title": "HR Test Doc",
            "page_number": 2,
            "section_id": "Section 24",
            "text": "Section 24 deals with deductions from income from house property.",
        },
    ]


def test_hybrid_search_returns_relevant_chunk_first():
    client = get_qdrant_client()
    ensure_collection(client)
    index_chunks_to_qdrant(client, _seed_chunks())

    results = hybrid_search(client, "What deduction does Section 80C provide?", top_k=10)
    assert len(results) > 0
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk_id == "hr-test-1"


def test_hybrid_search_respects_doc_type_filter():
    client = get_qdrant_client()
    results = hybrid_search(client, "deduction", top_k=10, doc_type_filter="Judgment")
    assert all(r.doc_type == "Judgment" for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hybrid_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval'`

- [ ] **Step 3: Create `retrieval/__init__.py`** (empty file)

- [ ] **Step 4: Write `retrieval/hybrid_retriever.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hybrid_retriever.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add retrieval/__init__.py retrieval/hybrid_retriever.py tests/test_hybrid_retriever.py
git commit -m "feat: hybrid dense+sparse retrieval with server-side RRF fusion"
```

---

### Task 4.2: Cross-encoder reranker

**Files:**
- Create: `retrieval/reranker.py`
- Create: `tests/test_reranker.py`

**Interfaces:**
- Consumes: `RetrievedChunk` (Task 4.1).
- Produces: `retrieval/reranker.py` exposes:
  - `get_reranker() -> CrossEncoder` — loads `BAAI/bge-reranker-large` (module-level cache).
  - `rerank(query: str, chunks: list[RetrievedChunk], top_k: int = 8) -> list[RetrievedChunk]` — re-scores `.score` in place with the cross-encoder score and returns the top-k sorted descending.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reranker.py
from retrieval.hybrid_retriever import RetrievedChunk
from retrieval.reranker import rerank


def _chunk(chunk_id, text):
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id="d", doc_type="Tax", title="t",
        page_number=1, section_id="s", text=text, score=0.0,
    )


def test_rerank_puts_more_relevant_chunk_first():
    chunks = [
        _chunk("c1", "The weather today is sunny with a chance of rain."),
        _chunk("c2", "Section 80C allows a tax deduction for life insurance premiums."),
    ]
    reranked = rerank("What deduction does Section 80C provide?", chunks, top_k=2)
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score > reranked[1].score


def test_rerank_respects_top_k():
    chunks = [_chunk(f"c{i}", f"chunk number {i} about tax law") for i in range(5)]
    reranked = rerank("tax law", chunks, top_k=3)
    assert len(reranked) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reranker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval.reranker'`

- [ ] **Step 3: Write `retrieval/reranker.py`**

```python
from functools import lru_cache

from sentence_transformers import CrossEncoder

from retrieval.hybrid_retriever import RetrievedChunk

_MODEL_NAME = "BAAI/bge-reranker-large"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(_MODEL_NAME)


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int = 8) -> list[RetrievedChunk]:
    if not chunks:
        return []

    model = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reranker.py -v`
Expected: PASS (downloads `bge-reranker-large` on first run, ~1.1GB)

- [ ] **Step 5: Commit**

```bash
git add retrieval/reranker.py tests/test_reranker.py
git commit -m "feat: cross-encoder reranking with bge-reranker-large"
```

---

### Task 4.3: Graph expander (1-hop citation pull-in)

**Files:**
- Create: `retrieval/graph_expander.py`
- Create: `tests/test_graph_expander.py`

**Interfaces:**
- Consumes: `RetrievedChunk` (4.1), `get_neo4j_driver` (1.3), `get_qdrant_client`/`COLLECTION_NAME` (1.2).
- Produces: `retrieval/graph_expander.py` exposes:
  - `expand_with_citations(driver, qdrant_client, chunks: list[RetrievedChunk], limit: int = 10) -> list[RetrievedChunk]` — for the given top chunks, finds `(:Chunk)-[:CITES]->(:Ref)<-[:DEFINED_IN]-(:Chunk)` 1-hop targets in Neo4j, fetches their full payload from Qdrant by `chunk_id`, deduplicates against already-present `chunk_id`s, caps additions at `limit`, and returns the original list plus the expansion appended (expanded chunks last, per the PRD's prompt-ordering guidance).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_expander.py
from indexing.graph_store import ensure_schema, get_neo4j_driver
from indexing.run_index import index_chunks_to_neo4j, index_chunks_to_qdrant
from indexing.vector_store import ensure_collection, get_qdrant_client
from retrieval.graph_expander import expand_with_citations
from retrieval.hybrid_retriever import RetrievedChunk


def _seed():
    return [
        {
            "chunk_id": "ge-statute-80c",
            "doc_id": "ge-doc",
            "doc_type": "Tax",
            "title": "GE Test Act",
            "page_number": 3,
            "section_id": "Section 80C",
            "text": "Section 80C allows deductions up to a specified limit.",
        },
        {
            "chunk_id": "ge-judgment-1",
            "doc_id": "ge-case",
            "doc_type": "Judgment",
            "title": "GE v. Test",
            "page_number": 1,
            "section_id": "para_1",
            "text": "The appellant claims relief under Section 80C.",
        },
    ]


def test_expand_with_citations_pulls_in_cited_section():
    qdrant_client = get_qdrant_client()
    ensure_collection(qdrant_client)
    index_chunks_to_qdrant(qdrant_client, _seed())

    driver = get_neo4j_driver()
    ensure_schema(driver)
    index_chunks_to_neo4j(driver, _seed())

    top_chunks = [
        RetrievedChunk(
            chunk_id="ge-judgment-1", doc_id="ge-case", doc_type="Judgment",
            title="GE v. Test", page_number=1, section_id="para_1",
            text="The appellant claims relief under Section 80C.", score=0.9,
        )
    ]

    expanded = expand_with_citations(driver, qdrant_client, top_chunks, limit=10)
    driver.close()

    chunk_ids = [c.chunk_id for c in expanded]
    assert "ge-judgment-1" in chunk_ids
    assert "ge-statute-80c" in chunk_ids
    assert chunk_ids[-1] == "ge-statute-80c"  # expanded chunks appended last


def test_expand_with_citations_deduplicates_already_present_chunks():
    driver = get_neo4j_driver()
    qdrant_client = get_qdrant_client()

    top_chunks = [
        RetrievedChunk(
            chunk_id="ge-statute-80c", doc_id="ge-doc", doc_type="Tax",
            title="GE Test Act", page_number=3, section_id="Section 80C",
            text="Section 80C allows deductions up to a specified limit.", score=0.95,
        ),
        RetrievedChunk(
            chunk_id="ge-judgment-1", doc_id="ge-case", doc_type="Judgment",
            title="GE v. Test", page_number=1, section_id="para_1",
            text="The appellant claims relief under Section 80C.", score=0.9,
        ),
    ]
    expanded = expand_with_citations(driver, qdrant_client, top_chunks, limit=10)
    driver.close()
    assert len(expanded) == 2  # no duplicate of ge-statute-80c added
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_expander.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval.graph_expander'`

- [ ] **Step 3: Write `retrieval/graph_expander.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_expander.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add retrieval/graph_expander.py tests/test_graph_expander.py
git commit -m "feat: 1-hop citation graph expansion"
```

---

### Task 4.4: Retrieval CLI harness

**Files:**
- Create: `retrieval/cli.py`
- Create: `tests/test_retrieval_cli.py`

**Interfaces:**
- Consumes: `hybrid_search` (4.1), `rerank` (4.2), `expand_with_citations` (4.3).
- Produces: `retrieval/cli.py` exposes:
  - `run_retrieval_pipeline(query: str, doc_type_filter: str | None = None) -> list[RetrievedChunk]` — full chain: hybrid_search(top_k=30) → rerank(top_k=8) → expand_with_citations(limit=10).
  - CLI: `python -m retrieval.cli "some query"` prints each result's `doc_id`, `page_number`, `section_id`, score, and a text snippet.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieval_cli.py
from retrieval.cli import run_retrieval_pipeline


def test_run_retrieval_pipeline_returns_ranked_chunks_with_expansion():
    results = run_retrieval_pipeline("What does Section 80C say about deductions?")
    assert len(results) > 0
    assert len(results) <= 18  # 8 reranked + up to 10 expanded
    assert results[0].score >= results[min(7, len(results) - 1)].score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval.cli'`

- [ ] **Step 3: Write `retrieval/cli.py`**

```python
import sys

from indexing.graph_store import get_neo4j_driver
from indexing.vector_store import get_qdrant_client
from retrieval.graph_expander import expand_with_citations
from retrieval.hybrid_retriever import RetrievedChunk, hybrid_search
from retrieval.reranker import rerank


def run_retrieval_pipeline(query: str, doc_type_filter: str | None = None) -> list[RetrievedChunk]:
    qdrant_client = get_qdrant_client()
    candidates = hybrid_search(qdrant_client, query, top_k=30, doc_type_filter=doc_type_filter)
    top = rerank(query, candidates, top_k=8)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval_cli.py -v`
Expected: PASS

- [ ] **Step 5: Manually sanity-check 5 sample queries against the real corpus**

Run each of:
```bash
python -m retrieval.cli "What medical expenses are deductible?"
python -m retrieval.cli "What tax credit exists for the elderly or disabled?"
python -m retrieval.cli "How does the Armed Forces tax guide treat combat pay?"
python -m retrieval.cli "What is Section 80C?"
python -m retrieval.cli "unrelated question about astrophysics"
```
Expected: the first four return plausible, on-topic top chunks; the fifth should show low scores across the board (a signal `insufficient_context` should later fire). Read the printed snippets and confirm they make sense — this is a manual judgment call, not an automated assertion.

- [ ] **Step 6: Commit**

```bash
git add retrieval/cli.py tests/test_retrieval_cli.py
git commit -m "feat: retrieval CLI harness for manual sanity-checking"
```

---

## Phase 5: Generation (Gemini Rotation, Prompts, Citation Guardrail)

**Goal of phase:** Turn a retrieved chunk set into a grounded, structured JSON answer with verified citations.

### Task 5.1: Gemini multi-key/model rotation wrapper

**Files:**
- Create: `generation/__init__.py`
- Create: `generation/gemini_rotation.py`
- Create: `tests/test_gemini_rotation.py`

**Interfaces:**
- Consumes: `config.settings.gemini_api_keys`, `config.settings.gemini_models`.
- Produces: `generation/gemini_rotation.py` exposes:
  - `class RateLimitError(Exception)`, `class AllPairsExhaustedError(Exception)`
  - `class GeminiRotator:` with `__init__(self, api_keys: list[str], models: list[str])`, method `generate(self, prompt: str, system_instruction: str, response_schema: dict, temperature: float = 0.1) -> str` (returns raw JSON text), private `_next_available_pair(self) -> tuple[str, str]`, private `_call(self, key: str, model: str, prompt: str, system_instruction: str, response_schema: dict, temperature: float) -> str`.
  - Cooldown: on a rate-limit exception from the underlying SDK, mark `(key, model)` in cooldown for 60s and retry the next pair; if all pairs are in cooldown, raise `AllPairsExhaustedError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gemini_rotation.py
import time
from unittest.mock import patch

import pytest

from generation.gemini_rotation import AllPairsExhaustedError, GeminiRotator, RateLimitError


def test_rotator_tries_next_pair_on_rate_limit():
    rotator = GeminiRotator(api_keys=["k1", "k2"], models=["m1"])
    call_log = []

    def fake_call(self, key, model, prompt, system_instruction, response_schema, temperature):
        call_log.append((key, model))
        if key == "k1":
            raise RateLimitError("429")
        return '{"answer": "ok"}'

    with patch.object(GeminiRotator, "_call", fake_call):
        result = rotator.generate("test prompt", "system", {}, temperature=0.1)

    assert result == '{"answer": "ok"}'
    assert call_log == [("k1", "m1"), ("k2", "m1")]


def test_rotator_raises_when_all_pairs_exhausted():
    rotator = GeminiRotator(api_keys=["k1"], models=["m1"])

    def always_fails(self, key, model, prompt, system_instruction, response_schema, temperature):
        raise RateLimitError("429")

    with patch.object(GeminiRotator, "_call", always_fails):
        with pytest.raises(AllPairsExhaustedError):
            rotator.generate("test prompt", "system", {}, temperature=0.1)


def test_rotator_skips_pair_in_cooldown():
    rotator = GeminiRotator(api_keys=["k1", "k2"], models=["m1"])
    rotator.cooldowns[("k1", "m1")] = time.time() + 60

    call_log = []

    def fake_call(self, key, model, prompt, system_instruction, response_schema, temperature):
        call_log.append((key, model))
        return '{"answer": "ok"}'

    with patch.object(GeminiRotator, "_call", fake_call):
        rotator.generate("test prompt", "system", {}, temperature=0.1)

    assert call_log == [("k2", "m1")]  # k1 skipped because it's in cooldown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gemini_rotation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generation'`

- [ ] **Step 3: Create `generation/__init__.py`** (empty file)

- [ ] **Step 4: Write `generation/gemini_rotation.py`**

```python
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable


class RateLimitError(Exception):
    pass


class ServerError(Exception):
    pass


class AllPairsExhaustedError(Exception):
    pass


class GeminiRotator:
    def __init__(self, api_keys: list[str], models: list[str]):
        self.pairs = [(k, m) for m in models for k in api_keys]
        self.cooldowns: dict[tuple[str, str], float] = {}
        self._idx = 0

    def _next_available_pair(self) -> tuple[str, str] | None:
        now = time.time()
        for _ in range(len(self.pairs)):
            pair = self.pairs[self._idx % len(self.pairs)]
            self._idx += 1
            unblock_at = self.cooldowns.get(pair)
            if unblock_at is None or unblock_at <= now:
                return pair
        return None

    def _call(
        self,
        key: str,
        model: str,
        prompt: str,
        system_instruction: str,
        response_schema: dict,
        temperature: float,
    ) -> str:
        genai.configure(api_key=key)
        gen_model = genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
        try:
            response = gen_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            return response.text
        except ResourceExhausted as e:
            raise RateLimitError(str(e)) from e
        except ServiceUnavailable as e:
            raise ServerError(str(e)) from e

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: dict,
        temperature: float = 0.1,
    ) -> str:
        attempts = 0
        while attempts < len(self.pairs):
            pair = self._next_available_pair()
            if pair is None:
                raise AllPairsExhaustedError("All (key, model) pairs are in cooldown.")
            key, model = pair
            try:
                return self._call(key, model, prompt, system_instruction, response_schema, temperature)
            except RateLimitError:
                self.cooldowns[pair] = time.time() + 60
                attempts += 1
                continue
            except ServerError:
                time.sleep(1)
                try:
                    return self._call(key, model, prompt, system_instruction, response_schema, temperature)
                except (RateLimitError, ServerError):
                    attempts += 1
                    continue
        raise AllPairsExhaustedError("All (key, model) pairs exhausted after retries.")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_gemini_rotation.py -v`
Expected: PASS (fully mocked, no real API calls)

- [ ] **Step 6: Commit**

```bash
git add generation/__init__.py generation/gemini_rotation.py tests/test_gemini_rotation.py
git commit -m "feat: Gemini multi-key/model rotation wrapper with cooldowns"
```

---

### Task 5.2: Prompt templates and structured output schema

**Files:**
- Create: `generation/prompts.py`
- Create: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `RetrievedChunk` (4.1).
- Produces: `generation/prompts.py` exposes:
  - `SYSTEM_PROMPT: str` (the fixed instruction prompt from the PRD, Section 11.1).
  - `RESPONSE_SCHEMA: dict` — Gemini `response_schema` JSON schema matching `{answer, citations: [{doc_name, page, excerpt_id}], confidence, insufficient_context}`.
  - `build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str` — numbered, tagged context blocks in the exact format from PRD Section 11.2, in the given chunk order (reranked first, graph-expanded last — this ordering is already guaranteed by `run_retrieval_pipeline` in Task 4.4).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
from generation.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from retrieval.hybrid_retriever import RetrievedChunk


def test_system_prompt_mentions_json_and_insufficient_context():
    assert "JSON" in SYSTEM_PROMPT
    assert "insufficient" in SYSTEM_PROMPT.lower()


def test_response_schema_has_required_fields():
    props = RESPONSE_SCHEMA["properties"]
    assert set(props.keys()) == {"answer", "citations", "confidence", "insufficient_context"}


def test_build_user_prompt_numbers_chunks_and_includes_metadata():
    chunks = [
        RetrievedChunk(
            chunk_id="c1", doc_id="Income Tax Act", doc_type="Act", title="Income Tax Act",
            page_number=42, section_id="Section 80C", text="Deduction text here.", score=0.9,
        )
    ]
    prompt = build_user_prompt("What is Section 80C?", chunks)
    assert "QUESTION:" in prompt
    assert "What is Section 80C?" in prompt
    assert "[1] Source: Income Tax Act, Page 42, Excerpt ID: c1" in prompt
    assert "Deduction text here." in prompt


def test_build_user_prompt_numbers_multiple_chunks_sequentially():
    chunks = [
        RetrievedChunk("c1", "d1", "Act", "Doc One", 1, "s1", "text one", 0.9),
        RetrievedChunk("c2", "d2", "Judgment", "Doc Two", 2, "s2", "text two", 0.8),
    ]
    prompt = build_user_prompt("q", chunks)
    assert "[1] Source: Doc One" in prompt
    assert "[2] Source: Doc Two" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generation.prompts'`

- [ ] **Step 3: Write `generation/prompts.py`**

```python
from retrieval.hybrid_retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a legal research assistant. You answer ONLY using the CONTEXT
provided below, which consists of numbered excerpts from Acts, Judgments,
Points of View, and Tax documents. You must not use outside knowledge.

Rules:
1. Every factual claim in your answer must be traceable to a specific
   excerpt in the CONTEXT.
2. If the CONTEXT does not contain enough information to answer, say so
   explicitly instead of guessing.
3. Cite excerpts inline using their [doc_name, page] tag exactly as given
   in the CONTEXT, e.g. [Income Tax Act, p.42].
4. Return ONLY valid JSON matching the schema below. No markdown, no
   preamble, no code fences.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_name": {"type": "string"},
                    "page": {"type": "integer"},
                    "excerpt_id": {"type": "string"},
                },
                "required": ["doc_name", "page", "excerpt_id"],
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "insufficient_context": {"type": "boolean"},
    },
    "required": ["answer", "citations", "confidence", "insufficient_context"],
}


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    lines = [f"QUESTION:\n{query}\n", "CONTEXT:"]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] Source: {chunk.doc_id}, Page {chunk.page_number}, Excerpt ID: {chunk.chunk_id}"
        )
        lines.append(chunk.text)
        lines.append("")
    lines.append(
        "Answer the QUESTION using only the CONTEXT above. Follow the system rules "
        "and return only the JSON object."
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add generation/prompts.py tests/test_prompts.py
git commit -m "feat: system prompt, response schema, and per-query prompt builder"
```

---

### Task 5.3: Citation guardrail

**Files:**
- Create: `generation/citation_guard.py`
- Create: `tests/test_citation_guard.py`

**Interfaces:**
- Consumes: `RetrievedChunk` (4.1).
- Produces: `generation/citation_guard.py` exposes:
  - `class GuardResult:` dataclass — `answer: str, citations: list[dict], confidence: str, insufficient_context: bool, all_citations_valid: bool`
  - `parse_gemini_json(raw: str) -> dict` — strips stray code fences (` ```json ... ``` `) before `json.loads`.
  - `verify_citations(parsed: dict, context_chunks: list[RetrievedChunk]) -> GuardResult` — checks each citation's `excerpt_id` against `{c.chunk_id for c in context_chunks}`; if any citation is invalid, drops it from the returned list and sets `insufficient_context = True` if zero valid citations remain, otherwise keeps the valid ones and leaves `insufficient_context` as returned by Gemini. Sets `all_citations_valid` to whether every original citation passed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_citation_guard.py
from generation.citation_guard import GuardResult, parse_gemini_json, verify_citations
from retrieval.hybrid_retriever import RetrievedChunk


def _context():
    return [
        RetrievedChunk("c1", "d1", "Act", "Income Tax Act", 42, "Section 80C", "text", 0.9),
    ]


def test_parse_gemini_json_strips_code_fences():
    raw = '```json\n{"answer": "hi", "citations": [], "confidence": "high", "insufficient_context": false}\n```'
    parsed = parse_gemini_json(raw)
    assert parsed["answer"] == "hi"


def test_parse_gemini_json_handles_plain_json():
    raw = '{"answer": "hi", "citations": [], "confidence": "high", "insufficient_context": false}'
    parsed = parse_gemini_json(raw)
    assert parsed["answer"] == "hi"


def test_verify_citations_passes_valid_citation():
    parsed = {
        "answer": "Section 80C allows a deduction [Income Tax Act, p.42].",
        "citations": [{"doc_name": "Income Tax Act", "page": 42, "excerpt_id": "c1"}],
        "confidence": "high",
        "insufficient_context": False,
    }
    result = verify_citations(parsed, _context())
    assert isinstance(result, GuardResult)
    assert result.all_citations_valid is True
    assert len(result.citations) == 1
    assert result.insufficient_context is False


def test_verify_citations_strips_invalid_citation_and_flags_insufficient():
    parsed = {
        "answer": "Some claim [Fake Doc, p.1].",
        "citations": [{"doc_name": "Fake Doc", "page": 1, "excerpt_id": "does-not-exist"}],
        "confidence": "high",
        "insufficient_context": False,
    }
    result = verify_citations(parsed, _context())
    assert result.all_citations_valid is False
    assert len(result.citations) == 0
    assert result.insufficient_context is True


def test_verify_citations_keeps_valid_and_drops_invalid_when_mixed():
    parsed = {
        "answer": "Mixed claim.",
        "citations": [
            {"doc_name": "Income Tax Act", "page": 42, "excerpt_id": "c1"},
            {"doc_name": "Fake Doc", "page": 1, "excerpt_id": "does-not-exist"},
        ],
        "confidence": "medium",
        "insufficient_context": False,
    }
    result = verify_citations(parsed, _context())
    assert result.all_citations_valid is False
    assert len(result.citations) == 1
    assert result.citations[0]["excerpt_id"] == "c1"
    assert result.insufficient_context is False  # at least one valid citation remains
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_citation_guard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generation.citation_guard'`

- [ ] **Step 3: Write `generation/citation_guard.py`**

```python
import json
import re
from dataclasses import dataclass

from retrieval.hybrid_retriever import RetrievedChunk

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class GuardResult:
    answer: str
    citations: list[dict]
    confidence: str
    insufficient_context: bool
    all_citations_valid: bool


def parse_gemini_json(raw: str) -> dict:
    cleaned = _CODE_FENCE_RE.sub("", raw).strip()
    return json.loads(cleaned)


def verify_citations(parsed: dict, context_chunks: list[RetrievedChunk]) -> GuardResult:
    valid_ids = {c.chunk_id for c in context_chunks}
    original_citations = parsed.get("citations", [])

    valid_citations = [c for c in original_citations if c.get("excerpt_id") in valid_ids]
    all_valid = len(valid_citations) == len(original_citations)

    insufficient_context = bool(parsed.get("insufficient_context", False))
    if not valid_citations and original_citations:
        insufficient_context = True

    return GuardResult(
        answer=parsed.get("answer", ""),
        citations=valid_citations,
        confidence=parsed.get("confidence", "low"),
        insufficient_context=insufficient_context,
        all_citations_valid=all_valid,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_citation_guard.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add generation/citation_guard.py tests/test_citation_guard.py
git commit -m "feat: citation guardrail — parse, verify, and strip invalid citations"
```

---

### Task 5.4: Generation orchestrator with one-shot regeneration

**Files:**
- Create: `generation/generate.py`
- Create: `tests/test_generate.py`

**Interfaces:**
- Consumes: `GeminiRotator` (5.1), `SYSTEM_PROMPT`/`RESPONSE_SCHEMA`/`build_user_prompt` (5.2), `parse_gemini_json`/`verify_citations`/`GuardResult` (5.3), `RetrievedChunk` (4.1).
- Produces: `generation/generate.py` exposes:
  - `get_rotator() -> GeminiRotator` — module-level singleton built from `config.settings`.
  - `generate_answer(query: str, context_chunks: list[RetrievedChunk]) -> GuardResult` — calls the rotator, parses+verifies; if `all_citations_valid` is False on the first attempt, regenerates once with a stricter reminder appended to the prompt ("Your previous answer cited a source not present in CONTEXT — only cite from the numbered excerpts above."), then verifies again and returns whichever result has more valid citations (or the second attempt if tied).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate.py
from unittest.mock import patch

from generation.generate import generate_answer
from generation.gemini_rotation import GeminiRotator
from retrieval.hybrid_retriever import RetrievedChunk


def _context():
    return [RetrievedChunk("c1", "d1", "Act", "Income Tax Act", 42, "Section 80C", "text", 0.9)]


def test_generate_answer_returns_valid_result_on_first_try():
    good_json = (
        '{"answer": "ok [Income Tax Act, p.42]", '
        '"citations": [{"doc_name": "Income Tax Act", "page": 42, "excerpt_id": "c1"}], '
        '"confidence": "high", "insufficient_context": false}'
    )
    with patch.object(GeminiRotator, "generate", return_value=good_json):
        result = generate_answer("What is Section 80C?", _context())

    assert result.all_citations_valid is True
    assert len(result.citations) == 1


def test_generate_answer_regenerates_once_on_invalid_citation():
    bad_json = (
        '{"answer": "bad", "citations": [{"doc_name": "Fake", "page": 1, "excerpt_id": "nope"}], '
        '"confidence": "high", "insufficient_context": false}'
    )
    good_json = (
        '{"answer": "fixed [Income Tax Act, p.42]", '
        '"citations": [{"doc_name": "Income Tax Act", "page": 42, "excerpt_id": "c1"}], '
        '"confidence": "high", "insufficient_context": false}'
    )
    call_count = {"n": 0}

    def fake_generate(self, prompt, system_instruction, response_schema, temperature=0.1):
        call_count["n"] += 1
        return bad_json if call_count["n"] == 1 else good_json

    with patch.object(GeminiRotator, "generate", fake_generate):
        result = generate_answer("What is Section 80C?", _context())

    assert call_count["n"] == 2
    assert result.all_citations_valid is True
    assert len(result.citations) == 1


def test_generate_answer_returns_best_effort_if_regeneration_still_invalid():
    bad_json = (
        '{"answer": "bad", "citations": [{"doc_name": "Fake", "page": 1, "excerpt_id": "nope"}], '
        '"confidence": "high", "insufficient_context": false}'
    )
    with patch.object(GeminiRotator, "generate", return_value=bad_json):
        result = generate_answer("What is Section 80C?", _context())

    assert result.all_citations_valid is False
    assert result.insufficient_context is True
    assert result.citations == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generation.generate'`

- [ ] **Step 3: Write `generation/generate.py`**

```python
from functools import lru_cache

from config import settings
from generation.citation_guard import GuardResult, parse_gemini_json, verify_citations
from generation.gemini_rotation import GeminiRotator
from generation.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from retrieval.hybrid_retriever import RetrievedChunk

_REMINDER = (
    "\n\nYour previous answer cited a source not present in CONTEXT — "
    "only cite from the numbered excerpts above."
)


@lru_cache(maxsize=1)
def get_rotator() -> GeminiRotator:
    return GeminiRotator(api_keys=settings.gemini_api_keys, models=settings.gemini_models)


def generate_answer(query: str, context_chunks: list[RetrievedChunk]) -> GuardResult:
    rotator = get_rotator()
    base_prompt = build_user_prompt(query, context_chunks)

    raw = rotator.generate(base_prompt, SYSTEM_PROMPT, RESPONSE_SCHEMA, temperature=0.1)
    parsed = parse_gemini_json(raw)
    result = verify_citations(parsed, context_chunks)

    if result.all_citations_valid:
        return result

    retry_prompt = base_prompt + _REMINDER
    raw_retry = rotator.generate(retry_prompt, SYSTEM_PROMPT, RESPONSE_SCHEMA, temperature=0.1)
    parsed_retry = parse_gemini_json(raw_retry)
    result_retry = verify_citations(parsed_retry, context_chunks)

    if len(result_retry.citations) >= len(result.citations):
        return result_retry
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generate.py -v`
Expected: PASS

- [ ] **Step 5: Test rotation behavior against a simulated 429 end-to-end**

Run a manual smoke test:
```python
# scratch check, not committed — run via `python -c`
from generation.gemini_rotation import GeminiRotator, RateLimitError
from unittest.mock import patch

rotator = GeminiRotator(api_keys=["real-or-fake-key-1", "real-or-fake-key-2"], models=["gemini-2.5-flash"])
call_count = {"n": 0}
def fake_call(self, key, model, prompt, system_instruction, response_schema, temperature):
    call_count["n"] += 1
    if call_count["n"] == 1:
        raise RateLimitError("simulated 429")
    return '{"answer": "recovered", "citations": [], "confidence": "low", "insufficient_context": true}'

with patch.object(GeminiRotator, "_call", fake_call):
    print(rotator.generate("test", "sys", {}, 0.1))
```
Expected: prints the recovered JSON, confirming the rotator moved to the second key after the first raised `RateLimitError`.

- [ ] **Step 6: Commit**

```bash
git add generation/generate.py tests/test_generate.py
git commit -m "feat: generation orchestrator with one-shot regeneration on invalid citations"
```

---

## Phase 6: FastAPI Backend + Frontend

**Goal of phase:** A working `/query` endpoint and a minimal page that exercises the full pipeline end-to-end in a browser.

### Task 6.1: Pydantic schemas and FastAPI endpoint

**Files:**
- Create: `app/__init__.py`
- Create: `app/schemas.py`
- Create: `app/api.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `run_retrieval_pipeline` (4.4, but called with explicit `doc_type_filter`/`top_k` params rather than the CLI defaults — see below), `generate_answer` (5.4), `RetrievedChunk` (4.1).
- Produces:
  - `app/schemas.py`: `QueryRequest(BaseModel)` — `query: str, doc_type_filter: str | None = None, top_k: int = 8`; `Citation(BaseModel)` — `doc_name: str, page: int, excerpt_id: str`; `QueryResponse(BaseModel)` — `answer: str, citations: list[Citation], confidence: str, insufficient_context: bool, retrieved_chunk_ids: list[str]`.
  - `app/api.py`: FastAPI `app` with CORS `allow_origins=["*"]`, `POST /query -> QueryResponse`, `GET /health -> {"status": "ok"}`.
  - Refactor note: Task 4.4's `run_retrieval_pipeline(query, doc_type_filter)` hardcodes `top_k=8` for rerank internally; the API's `req.top_k` is passed through to the rerank step, so `run_retrieval_pipeline` gains an optional `rerank_top_k: int = 8` parameter (update Task 4.4's signature — this is additive and keeps the existing CLI call working via the default).

- [ ] **Step 1: Update `retrieval/cli.py`'s `run_retrieval_pipeline` to accept `rerank_top_k`**

```python
# In retrieval/cli.py, change the signature and body of run_retrieval_pipeline to:
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
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_api.py
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app
from generation.citation_guard import GuardResult
from retrieval.hybrid_retriever import RetrievedChunk

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_answer_and_citations():
    fake_chunks = [RetrievedChunk("c1", "d1", "Act", "Income Tax Act", 42, "Section 80C", "text", 0.9)]
    fake_result = GuardResult(
        answer="Section 80C allows a deduction [Income Tax Act, p.42].",
        citations=[{"doc_name": "Income Tax Act", "page": 42, "excerpt_id": "c1"}],
        confidence="high",
        insufficient_context=False,
        all_citations_valid=True,
    )

    with patch("app.api.run_retrieval_pipeline", return_value=fake_chunks), \
         patch("app.api.generate_answer", return_value=fake_result):
        response = client.post("/query", json={"query": "What is Section 80C?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == fake_result.answer
    assert data["citations"] == [{"doc_name": "Income Tax Act", "page": 42, "excerpt_id": "c1"}]
    assert data["confidence"] == "high"
    assert data["insufficient_context"] is False
    assert data["retrieved_chunk_ids"] == ["c1"]


def test_query_endpoint_passes_doc_type_filter_and_top_k():
    with patch("app.api.run_retrieval_pipeline", return_value=[]) as mock_pipeline, \
         patch("app.api.generate_answer", return_value=GuardResult("", [], "low", True, True)):
        client.post("/query", json={"query": "q", "doc_type_filter": "Act", "top_k": 5})

    mock_pipeline.assert_called_once_with("q", doc_type_filter="Act", rerank_top_k=5)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 4: Create `app/__init__.py`** (empty file)

- [ ] **Step 5: Write `app/schemas.py`**

```python
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    doc_type_filter: str | None = None
    top_k: int = 8


class Citation(BaseModel):
    doc_name: str
    page: int
    excerpt_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str
    insufficient_context: bool
    retrieved_chunk_ids: list[str]
```

- [ ] **Step 6: Write `app/api.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import QueryRequest, QueryResponse
from generation.generate import generate_answer
from retrieval.cli import run_retrieval_pipeline

app = FastAPI(title="Legal RAG API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    chunks = run_retrieval_pipeline(req.query, doc_type_filter=req.doc_type_filter, rerank_top_k=req.top_k)
    result = generate_answer(req.query, chunks)
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
        insufficient_context=result.insufficient_context,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add app/__init__.py app/schemas.py app/api.py tests/test_api.py retrieval/cli.py
git commit -m "feat: FastAPI /query and /health endpoints"
```

---

### Task 6.2: Frontend (single-page HTML/JS)

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/style.css`
- Create: `frontend/app.js`

**Interfaces:**
- Consumes: `POST /query` (Task 6.1), assumed served at `http://localhost:8000`.
- No Python interfaces — this is a static asset task, verified manually in a browser rather than with pytest.

- [ ] **Step 1: Write `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Legal RAG Search</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <div id="app">
    <h1>Legal RAG Search</h1>
    <div class="controls">
      <select id="docType">
        <option value="">All document types</option>
        <option value="Act">Act</option>
        <option value="Judgment">Judgment</option>
        <option value="POV">POV</option>
        <option value="Tax">Tax</option>
      </select>
      <input id="q" placeholder="Ask a legal question..." />
      <button id="ask">Ask</button>
    </div>
    <div id="status"></div>
    <div id="insufficientBanner" class="banner hidden">
      Insufficient context in corpus for a confident answer.
    </div>
    <div id="answer"></div>
    <ul id="citations"></ul>
  </div>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `frontend/style.css`**

```css
body {
  font-family: system-ui, sans-serif;
  max-width: 720px;
  margin: 2rem auto;
  padding: 0 1rem;
  color: #1e1e1e;
}

.controls {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

#q {
  flex: 1;
  padding: 0.5rem;
}

button {
  padding: 0.5rem 1rem;
  cursor: pointer;
}

.banner {
  background: #fff3cd;
  border: 1px solid #ffe69c;
  padding: 0.75rem;
  border-radius: 4px;
  margin-bottom: 1rem;
}

.banner.hidden {
  display: none;
}

#answer {
  white-space: pre-wrap;
  line-height: 1.5;
  margin-bottom: 1rem;
}

#citations li {
  margin-bottom: 0.25rem;
}
```

- [ ] **Step 3: Write `frontend/app.js`**

```javascript
const API = "http://localhost:8000";

const statusEl = document.getElementById("status");
const bannerEl = document.getElementById("insufficientBanner");
const answerEl = document.getElementById("answer");
const citationsEl = document.getElementById("citations");
const askButton = document.getElementById("ask");

askButton.onclick = async () => {
  const query = document.getElementById("q").value.trim();
  if (!query) return;

  const doc_type_filter = document.getElementById("docType").value || null;

  askButton.disabled = true;
  statusEl.textContent = "Searching...";
  bannerEl.classList.add("hidden");
  answerEl.textContent = "";
  citationsEl.innerHTML = "";

  try {
    const res = await fetch(`${API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, doc_type_filter, top_k: 8 }),
    });

    if (!res.ok) {
      statusEl.textContent = `Request failed: ${res.status}`;
      return;
    }

    const data = await res.json();

    statusEl.textContent = data.insufficient_context
      ? ""
      : `Confidence: ${data.confidence}`;
    bannerEl.classList.toggle("hidden", !data.insufficient_context);
    answerEl.textContent = data.answer;
    citationsEl.innerHTML = data.citations
      .map((c) => `<li>${c.doc_name} — p.${c.page}</li>`)
      .join("");
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    askButton.disabled = false;
  }
};
```

- [ ] **Step 4: Start the backend and frontend, and manually test end-to-end**

Run in one terminal:
```bash
uvicorn app.api:app --reload --port 8000
```
Run in another terminal:
```bash
cd frontend && python -m http.server 5500
```
Open `http://localhost:5500` in a browser, select "Tax" from the dropdown, type a question like "What medical expenses are deductible?", click Ask.
Expected: a loading state appears, then an answer with inline citations and a citations list appear below. Try an out-of-corpus question (e.g. "What is the capital of France?") and confirm the insufficient-context banner appears.

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: minimal HTML/JS frontend for the query UI"
```

---

## Phase 7: Evaluation

**Goal of phase:** A golden question set and a Ragas report measuring context recall, context precision, faithfulness, and answer relevancy, runnable after any pipeline change.

### Task 7.1: Golden question set

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/golden_set.json`
- Create: `tests/test_golden_set.py`

**Interfaces:**
- Produces: `evaluation/golden_set.json` — a JSON array of objects: `{"query": str, "ground_truth_answer": str, "ground_truth_doc_id": str, "ground_truth_page": int, "expect_insufficient_context": bool}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_golden_set.py
import json


def test_golden_set_has_required_fields_and_covers_out_of_corpus_cases():
    with open("evaluation/golden_set.json") as f:
        data = json.load(f)

    assert len(data) >= 10
    required_fields = {"query", "ground_truth_answer", "ground_truth_doc_id", "ground_truth_page", "expect_insufficient_context"}
    for item in data:
        assert required_fields.issubset(item.keys())

    out_of_corpus = [item for item in data if item["expect_insufficient_context"]]
    assert len(out_of_corpus) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_golden_set.py -v`
Expected: FAIL — `evaluation/golden_set.json` does not exist

- [ ] **Step 3: Create `evaluation/__init__.py`** (empty file)

- [ ] **Step 4: Write `evaluation/golden_set.json`**

Author 15–20 question/answer pairs by hand, reading the actual PDFs in `PDFS/` to find real answers and page numbers (this cannot be templated — it requires reading the corpus). Include at least 2 questions with `expect_insufficient_context: true` that are clearly outside the corpus's topic (e.g. unrelated general-knowledge questions). Example starting structure:

```json
[
  {
    "query": "What medical expenses can be deducted according to Publication 502?",
    "ground_truth_answer": "Medical expenses include payments for diagnosis, cure, mitigation, treatment, or prevention of disease, and for treatments affecting any part or function of the body.",
    "ground_truth_doc_id": "IRS Publication 502 Medical and dental expenses",
    "ground_truth_page": 1,
    "expect_insufficient_context": false
  },
  {
    "query": "What is the recipe for chocolate chip cookies?",
    "ground_truth_answer": "",
    "ground_truth_doc_id": "",
    "ground_truth_page": 0,
    "expect_insufficient_context": true
  }
]
```
Expand this to 15–20 entries covering all document types present in `PDFS/` at the time of evaluation — read each PDF's actual content to write accurate ground truths and page numbers.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_golden_set.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add evaluation/__init__.py evaluation/golden_set.json tests/test_golden_set.py
git commit -m "feat: golden question set for evaluation"
```

---

### Task 7.2: Ragas evaluation runner

**Files:**
- Create: `evaluation/run_eval.py`
- Create: `tests/test_run_eval.py`

**Interfaces:**
- Consumes: `evaluation/golden_set.json` (7.1), `run_retrieval_pipeline` (4.4/6.1), `generate_answer` (5.4).
- Produces: `evaluation/run_eval.py` exposes:
  - `load_golden_set(path: str) -> list[dict]`
  - `run_pipeline_for_eval(query: str) -> dict` — runs retrieval+generation, returns `{"answer": str, "contexts": list[str]}`.
  - `build_eval_rows(golden_set: list[dict]) -> list[dict]` — for each item, calls `run_pipeline_for_eval` and assembles a Ragas-ready row: `{"question", "answer", "contexts", "ground_truth"}`.
  - CLI entry: `python -m evaluation.run_eval` builds rows, runs `ragas.evaluate` with `context_recall, context_precision, faithfulness, answer_relevancy`, writes `evaluation/report.csv`, and prints the aggregate scores.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_eval.py
from unittest.mock import patch

from evaluation.run_eval import build_eval_rows, load_golden_set, run_pipeline_for_eval
from retrieval.hybrid_retriever import RetrievedChunk
from generation.citation_guard import GuardResult


def test_load_golden_set_reads_json():
    data = load_golden_set("evaluation/golden_set.json")
    assert len(data) >= 10


def test_run_pipeline_for_eval_returns_answer_and_contexts():
    fake_chunks = [RetrievedChunk("c1", "d1", "Tax", "Doc", 1, "s1", "chunk text", 0.9)]
    fake_result = GuardResult("the answer", [], "high", False, True)

    with patch("evaluation.run_eval.run_retrieval_pipeline", return_value=fake_chunks), \
         patch("evaluation.run_eval.generate_answer", return_value=fake_result):
        out = run_pipeline_for_eval("some query")

    assert out["answer"] == "the answer"
    assert out["contexts"] == ["chunk text"]


def test_build_eval_rows_produces_ragas_ready_rows():
    golden_set = [
        {"query": "q1", "ground_truth_answer": "gt1", "ground_truth_doc_id": "d1", "ground_truth_page": 1, "expect_insufficient_context": False}
    ]
    fake_out = {"answer": "a1", "contexts": ["ctx1"]}

    with patch("evaluation.run_eval.run_pipeline_for_eval", return_value=fake_out):
        rows = build_eval_rows(golden_set)

    assert rows == [{"question": "q1", "answer": "a1", "contexts": ["ctx1"], "ground_truth": "gt1"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.run_eval'`

- [ ] **Step 3: Write `evaluation/run_eval.py`**

```python
import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from generation.generate import generate_answer
from retrieval.cli import run_retrieval_pipeline

GOLDEN_SET_PATH = "evaluation/golden_set.json"
REPORT_PATH = "evaluation/report.csv"


def load_golden_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_pipeline_for_eval(query: str) -> dict:
    chunks = run_retrieval_pipeline(query)
    result = generate_answer(query, chunks)
    return {"answer": result.answer, "contexts": [c.text for c in chunks]}


def build_eval_rows(golden_set: list[dict]) -> list[dict]:
    rows = []
    for item in golden_set:
        out = run_pipeline_for_eval(item["query"])
        rows.append(
            {
                "question": item["query"],
                "answer": out["answer"],
                "contexts": out["contexts"],
                "ground_truth": item["ground_truth_answer"],
            }
        )
    return rows


if __name__ == "__main__":
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    rows = build_eval_rows(golden_set)
    report = evaluate(
        Dataset.from_list(rows),
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
    )
    report.to_pandas().to_csv(REPORT_PATH)
    print(report)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_eval.py -v`
Expected: PASS

- [ ] **Step 5: Run the full evaluation against the real corpus**

Run: `python -m evaluation.run_eval`
Expected: prints aggregate scores for context_recall, context_precision, faithfulness, answer_relevancy, and writes `evaluation/report.csv`. This step calls Gemini for every golden-set question, so it will consume rotation quota — expect it to take a few minutes for 15–20 questions. Compare the printed scores against the target thresholds in `specs/design.md` (Context Recall ≥ 0.85, Faithfulness ≥ 0.90); if either falls short, that's a signal to revisit chunking (Task 2.3) or reranking (Task 4.2) before considering the project done — but that revisiting is a separate follow-up task, not part of this plan.

- [ ] **Step 6: Commit**

```bash
git add evaluation/run_eval.py tests/test_run_eval.py
git commit -m "feat: Ragas evaluation runner producing evaluation/report.csv"
```

---

## Summary of files created

```
legal-rag/
├── .env.example
├── .gitignore
├── requirements.txt
├── config.py
├── specs/
│   ├── design.md
│   └── plan.md
├── ingestion/
│   ├── __init__.py
│   ├── parser.py
│   ├── classifier.py
│   ├── chunker.py
│   ├── run_ingest.py
│   └── chunks.json          # generated, not hand-written
├── indexing/
│   ├── __init__.py
│   ├── vector_store.py
│   ├── graph_store.py
│   ├── embedder.py
│   ├── sparse_encoder.py
│   ├── citation_extractor.py
│   └── run_index.py
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── graph_expander.py
│   └── cli.py
├── generation/
│   ├── __init__.py
│   ├── gemini_rotation.py
│   ├── prompts.py
│   ├── citation_guard.py
│   └── generate.py
├── evaluation/
│   ├── __init__.py
│   ├── golden_set.json
│   ├── run_eval.py
│   └── report.csv            # generated, not hand-written
├── app/
│   ├── __init__.py
│   ├── schemas.py
│   └── api.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
│   └── (one test file per module above)
└── PDFS/                      # existing corpus, untouched
```
