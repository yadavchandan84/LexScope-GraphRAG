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

    @staticmethod
    def _require(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value

    @staticmethod
    def _optional(name: str) -> str:
        return (os.environ.get(name) or "").strip()


settings = Settings()
