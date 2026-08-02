import logging
from typing import List, Union
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Auth & Database
    JWT_SECRET: str
    JWT_ALGORITHM: str
    MONGO_URI: str
    MONGO_DB_NAME: str
    REDIS_URL: str
    GEMINI_API_KEY: str
    ALLOWED_ORIGINS: str

    # Logging
    LOG_LEVEL: str = "INFO"

    # Qdrant Cloud & Vector Search Configuration
    QDRANT_URL: str  
    QDRANT_API_KEY: str  
    EMBEDDING_MODEL_NAME: str
    AVAILABLE_COLLECTIONS: Union[str, List[str]]
    HF_TOKEN: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @computed_field
    @property
    def parsed_origins(self) -> list[str]:
        if not self.ALLOWED_ORIGINS or self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @computed_field
    @property
    def parsed_collections(self) -> list[str]:
        if isinstance(self.AVAILABLE_COLLECTIONS, list):
            return self.AVAILABLE_COLLECTIONS
        if isinstance(self.AVAILABLE_COLLECTIONS, str):
            return [c.strip() for c in self.AVAILABLE_COLLECTIONS.split(",") if c.strip()]


# Instantiate settings (will validate required env vars)
try:
    settings = Settings()
except Exception as e:
    # pydantic-settings validation errors list missing/invalid field names only,
    # never the values, so this is safe to log without leaking secrets.
    logger.error("Failed to load application configuration: %s", e)
    raise e
