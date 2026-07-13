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
