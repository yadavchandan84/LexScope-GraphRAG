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
