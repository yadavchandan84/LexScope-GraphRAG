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
    fn = filename.lower()

    # Strong content signals take precedence over filename hints, so a file named
    # "income_tax_act.pdf" whose body reads like an Act is classified as an Act.
    if _matches_any(sample_text, _JUDGMENT_PATTERNS):
        return DocType.JUDGMENT
    if _matches_any(sample_text, _ACT_PATTERNS):
        return DocType.ACT
    if _matches_any(sample_text, _TAX_PATTERNS):
        return DocType.TAX

    # Fall back to filename hints when the text is inconclusive.
    if "judgment" in fn:
        return DocType.JUDGMENT
    if "act" in fn:
        return DocType.ACT
    if "tax" in fn or "irs" in fn or "publication" in fn:
        return DocType.TAX
    return DocType.POV


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)
