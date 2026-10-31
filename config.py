import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        # Qdrant is optional: when QDRANT_URL is set we use Qdrant Cloud, otherwise
        # we fall back to an on-disk embedded Qdrant at QDRANT_LOCAL_PATH (no server).
        self.qdrant_url = self._optional("QDRANT_URL")
        self.qdrant_api_key = self._optional("QDRANT_API_KEY")
        self.qdrant_local_path = self._optional("QDRANT_LOCAL_PATH") or "qdrant_data"
        self.neo4j_uri = self._require("NEO4J_URI")
        # Aura's downloaded environment template calls this NEO4J_USERNAME,
        # while some local setups use NEO4J_USER. Support either spelling.
        self.neo4j_user = self._optional("NEO4J_USER") or self._require("NEO4J_USERNAME")
        self.neo4j_password = self._require("NEO4J_PASSWORD")
        self.gemini_api_keys = [
            k.strip() for k in self._require("GEMINI_API_KEYS").split(",") if k.strip()
        ]
        self.gemini_models = [
            m.strip() for m in self._require("GEMINI_MODELS").split(",") if m.strip()
        ]

        # --- Caching (optional) ---------------------------------------------
        # A Redis URL enables a shared, persistent cache for answers and query
        # embeddings. When REDIS_URL is unset (or Redis is unreachable) we fall
        # back to an in-process cache so the app still works with no extra infra.
        self.redis_url = self._optional("REDIS_URL")
        self.cache_enabled = self._optional("CACHE_ENABLED").lower() not in ("0", "false", "no")
        # TTLs in seconds. Answers change only when the corpus is re-indexed, so
        # they can live a long time; embeddings are deterministic per model.
        self.cache_answer_ttl = self._optional_int("CACHE_ANSWER_TTL", 60 * 60 * 24 * 7)
        self.cache_embedding_ttl = self._optional_int("CACHE_EMBEDDING_TTL", 60 * 60 * 24 * 30)

        # --- Timeouts (seconds) ---------------------------------------------
        self.gemini_timeout = self._optional_int("GEMINI_TIMEOUT", 60)
        self.neo4j_timeout = self._optional_int("NEO4J_TIMEOUT", 15)

        # --- Models ----------------------------------------------------------
        # Defaults are the small, CPU/RAM-friendly variants so the API fits on
        # modest hosts (e.g. Render 512MB-2GB). For best quality on a big box,
        # switch to BAAI/bge-large-en-v1.5 (dim 1024) + BAAI/bge-reranker-large.
        # IMPORTANT: EMBEDDING_DIM must match EMBEDDING_MODEL's output size, and
        # changing the embedder requires re-indexing the corpus.
        self.embedding_model = self._optional("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
        self.embedding_dim = self._optional_int("EMBEDDING_DIM", 384)
        self.reranker_model = self._optional("RERANKER_MODEL") or "BAAI/bge-reranker-base"
        # When false, skip the cross-encoder reranker entirely (lowest RAM).
        self.reranker_enabled = self._optional("RERANKER_ENABLED").lower() not in ("0", "false", "no")

    @staticmethod
    def _require(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value

    @staticmethod
    def _optional(name: str) -> str:
        return (os.environ.get(name) or "").strip()

    @staticmethod
    def _optional_int(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default


settings = Settings()
