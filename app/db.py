from contextlib import asynccontextmanager
import motor.motor_asyncio
import redis.asyncio as redis
from fastapi import FastAPI
from app.config import settings


class DatabaseContainer:
    mongo_client: motor.motor_asyncio.AsyncIOMotorClient | None = None
    redis_client: redis.Redis | None = None


db_container = DatabaseContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for connecting and disconnecting
    MongoDB (Motor) and Redis async clients.
    """
    # Startup: Connect to MongoDB and Redis
    db_container.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
        settings.MONGO_URI
    )
    db_container.redis_client = redis.from_url(
        settings.REDIS_URL, decode_responses=True
    )

    yield

    # Shutdown: Close connections
    if db_container.mongo_client:
        db_container.mongo_client.close()
    if db_container.redis_client:
        await db_container.redis_client.aclose()


def get_db() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    """Dependency accessor for MongoDB database instance."""
    if db_container.mongo_client is None:
        raise RuntimeError("Database connection is not initialized.")
    return db_container.mongo_client[settings.MONGO_DB_NAME]


def get_redis() -> redis.Redis:
    """Dependency accessor for Redis client instance."""
    if db_container.redis_client is None:
        raise RuntimeError("Redis connection is not initialized.")
    return db_container.redis_client
