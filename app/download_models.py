"""Download all runtime models into MODEL_CACHE_DIR.

Run during the build (Render / Docker) so the service starts without fetching
models over the network:

    python -m app.download_models

No database or Gemini credentials are needed for this step.
"""

import os

# config.py requires these at import time, but downloading models doesn't use
# them. Placeholders let the build run before (or without) real secrets.
for _name in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GEMINI_API_KEYS", "GEMINI_MODELS"):
    os.environ.setdefault(_name, "unused-during-build")

from config import settings  # noqa: E402
from indexing.embedder import get_embedder  # noqa: E402
from indexing.sparse_encoder import get_sparse_model  # noqa: E402


def main() -> None:
    print(f"Downloading models into '{settings.model_cache_dir}'...")
    get_embedder()
    print(f"  dense:  {settings.embedding_model}")
    get_sparse_model()
    print("  sparse: Qdrant/bm25")
    # Always fetch the reranker so it can be switched on later with only an
    # env var change (it's ~23 MB).
    from retrieval.reranker import get_reranker

    get_reranker()
    print(f"  rerank: {settings.reranker_model}")
    print("Done.")


if __name__ == "__main__":
    main()
