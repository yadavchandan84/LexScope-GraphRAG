"""Cross-encoder reranking on ONNX Runtime (no PyTorch).

Loads an ONNX cross-encoder plus its tokenizer from the Hugging Face Hub. The
default, Xenova/ms-marco-MiniLM-L-6-v2 (quantized, ~23 MB), is small enough for
the Render free plan. Set RERANKER_ENABLED=false to skip reranking entirely and
keep the hybrid-search (RRF) order.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from config import settings
from retrieval.hybrid_retriever import RetrievedChunk

_MODEL_NAME = settings.reranker_model
_BATCH_SIZE = 4  # small batches keep peak RAM low on 512 MB hosts


class OnnxCrossEncoder:
    def __init__(self, repo_id: str, onnx_file: str, max_length: int, cache_dir: str):
        model_path = hf_hub_download(repo_id, f"onnx/{onnx_file}", cache_dir=cache_dir)
        tokenizer_path = hf_hub_download(repo_id, "tokenizer.json", cache_dir=cache_dir)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding()

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # Render free plan has 0.1 CPU
        options.inter_op_num_threads = 1
        # The default memory arena keeps the largest batch's buffers forever
        # (~130 MB here). Freeing them after each run keeps RAM flat.
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def predict(self, query: str, passages: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(passages), _BATCH_SIZE):
            batch = passages[start:start + _BATCH_SIZE]
            encodings = self.tokenizer.encode_batch([(query, p) for p in batch])
            feeds = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            }
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.array(
                    [e.type_ids for e in encodings], dtype=np.int64
                )
            logits = self.session.run(None, feeds)[0]
            scores.extend(np.asarray(logits).reshape(len(batch), -1)[:, 0].tolist())
        return scores


@lru_cache(maxsize=1)
def get_reranker() -> OnnxCrossEncoder:
    return OnnxCrossEncoder(
        _MODEL_NAME,
        settings.reranker_onnx_file,
        settings.reranker_max_length,
        settings.model_cache_dir,
    )


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int = 8) -> list[RetrievedChunk]:
    if not chunks:
        return []

    # When reranking is disabled, keep the hybrid-search order (already RRF-fused).
    if not settings.reranker_enabled:
        return chunks[:top_k]

    scores = get_reranker().predict(query, [c.text for c in chunks])
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked[:top_k]
