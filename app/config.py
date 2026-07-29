from typing import List, Union
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Auth & Database
    JWT_SECRET: str = "your_shared_jwt_secret_key_here"
    JWT_ALGORITHM: str = "HS256"
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "chatbot_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    ANTHROPIC_API_KEY: str = ""
    ALLOWED_ORIGINS: str = "*"

    # Qdrant Cloud & Vector Search Configuration
    QDRANT_URL: str  # Required, no default - cloud cluster URL
    QDRANT_API_KEY: str  # Required, no default - cloud API key
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    AVAILABLE_COLLECTIONS: Union[str, List[str]] = ["gita_collection", "mahabharata_collection"]

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
        return ["gita_collection", "mahabharata_collection"]


# Instantiate settings (will validate required env vars)
try:
    settings = Settings()
except Exception as e:
    # Print a helpful startup error if QDRANT_URL or QDRANT_API_KEY are missing
    print(f"\n[CONFIG ERROR] Failed to load application configuration: {e}")
    print("Please set QDRANT_URL and QDRANT_API_KEY in your .env file.\n")
    raise e
