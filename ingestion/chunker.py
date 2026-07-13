import hashlib
import re
import uuid
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
    # Deterministic (content-addressed) UUID: Qdrant's local mode requires point
    # IDs to be an unsigned int or a valid UUID, and this same id is reused as the
    # Neo4j chunk_id and the graph-expander lookup key, so keep one canonical form.
    digest = hashlib.sha256(f"{doc_id}|{section_id}|{text}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=digest[:32]))


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
