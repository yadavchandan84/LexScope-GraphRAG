from retrieval.hybrid_retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a legal research assistant. You answer ONLY using the CONTEXT
provided below, which consists of numbered excerpts from Acts, Judgments,
Points of View, and Tax documents. You must not use outside knowledge.

Rules:
1. Every factual claim in your answer must be traceable to a specific
   excerpt in the CONTEXT.
2. If the CONTEXT does not contain enough information to answer, do not
   guess: set "insufficient_context" to true and say so explicitly.
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
            f"[{i}] Source: {chunk.title}, Page {chunk.page_number}, Excerpt ID: {chunk.chunk_id}"
        )
        lines.append(chunk.text)
        lines.append("")
    lines.append(
        "Answer the QUESTION using only the CONTEXT above. Follow the system rules "
        "and return only the JSON object."
    )
    return "\n".join(lines)
