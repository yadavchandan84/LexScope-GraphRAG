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
        # All models run on ONNX Runtime (no PyTorch), so the API fits in the
        # 512 MB Render free plan.
        # Embeddings use fastembed. EMBEDDING_DIM must match the model, and
        # changing the embedder requires re-indexing the corpus:
        #   BAAI/bge-small-en-v1.5 -> 384 (default)
        #   BAAI/bge-base-en-v1.5  -> 768 (better quality, more RAM)
        self.embedding_model = self._optional("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
        self.embedding_dim = self._optional_int("EMBEDDING_DIM", 384)
        # Reranker: a Hugging Face repo that ships onnx/<RERANKER_ONNX_FILE> and
        # tokenizer.json. The default MiniLM cross-encoder is ~23 MB quantized.
        self.reranker_model = self._optional("RERANKER_MODEL") or "Xenova/ms-marco-MiniLM-L-6-v2"
        self.reranker_onnx_file = self._optional("RERANKER_ONNX_FILE") or "model_quantized.onnx"
        self.reranker_max_length = self._optional_int("RERANKER_MAX_LENGTH", 256)
        # When false, skip the cross-encoder and keep the hybrid (RRF) order.
        self.reranker_enabled = self._optional("RERANKER_ENABLED").lower() not in ("0", "false", "no")
        # Where downloaded models are stored. The Render build downloads them
        # into this folder so the running service doesn't fetch them on start.
        self.model_cache_dir = self._optional("MODEL_CACHE_DIR") or "models_cache"
        # ONNX Runtime threads for the embedding models. Unset = all cores
        # (fastest for local indexing); 1 = least RAM (used on Render).
        threads = self._optional_int("ONNX_THREADS", 0)
        self.onnx_threads = threads if threads > 0 else None

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
