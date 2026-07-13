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
