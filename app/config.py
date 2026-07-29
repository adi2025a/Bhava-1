from typing import List, Union
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Auth & Database
    JWT_SECRET: str
    JWT_ALGORITHM: str
    MONGO_URI: str
    MONGO_DB_NAME: str
    REDIS_URL: str
    ANTHROPIC_API_KEY: str
    ALLOWED_ORIGINS: str

    # Qdrant Cloud & Vector Search Configuration
    QDRANT_URL: str  
    QDRANT_API_KEY: str  
    EMBEDDING_MODEL_NAME: str
    AVAILABLE_COLLECTIONS: Union[str, List[str]]

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
    # Print a helpful startup error if QDRANT_URL or QDRANT_API_KEY are missing
    print(f"\n[CONFIG ERROR] Failed to load application configuration: {e}")
    print("Please set QDRANT_URL and QDRANT_API_KEY in your .env file.\n")
    raise e
