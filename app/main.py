from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import db_container
import motor.motor_asyncio
import redis.asyncio as redis

from app.routes.health import router as health_router
from app.routes.chat import router as chat_router
from app.services.embeddings import load_embedding_model
from app.services.retrieval import get_qdrant_client


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """
    FastAPI Lifespan event handler:
    1. Connects to MongoDB and Redis async clients.
    2. Warm loads SentenceTransformer embedding model into memory.
    3. Verifies Qdrant Cloud connection and credentials.
    """
    print("\n[STARTUP] Starting FastAPI Chatbot RAG Microservice...")

    # 1. Connect MongoDB & Redis
    print(f"[STARTUP] Connecting to MongoDB ({settings.MONGO_URI}) and Redis ({settings.REDIS_URL})...")
    db_container.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGO_URI)
    db_container.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

    # 2. Warm load embedding model once at startup
    print(f"[STARTUP] Warm-loading embedding model '{settings.EMBEDDING_MODEL_NAME}'...")
    try:
        load_embedding_model()
    except Exception as e:
        print(f"[STARTUP ERROR] Failed to load embedding model: {e}")
        raise e

    # 3. Verify Qdrant Cloud connection & credentials
    print(f"[STARTUP] Verifying Qdrant Cloud connectivity ({settings.QDRANT_URL})...")
    try:
        q_client = get_qdrant_client()
        collections_resp = q_client.get_collections()
        existing = [c.name for c in collections_resp.collections]
        print(f"[STARTUP] Qdrant Cloud connection verified. Existing collections: {existing}")
    except Exception as e:
        print(f"[STARTUP ERROR] Unable to connect to Qdrant Cloud. Check QDRANT_URL and QDRANT_API_KEY. Error: {e}")
        raise RuntimeError(f"Qdrant Cloud connection error: {e}") from e

    print("[STARTUP] Microservice startup complete! Ready to accept requests.\n")

    yield

    # Shutdown: Close connections
    print("\n[SHUTDOWN] Closing database & cache connections...")
    if db_container.mongo_client:
        db_container.mongo_client.close()
    if db_container.redis_client:
        await db_container.redis_client.aclose()
    print("[SHUTDOWN] Microservice shutdown complete.")


app = FastAPI(
    title="FastAPI Chatbot RAG Microservice",
    description="RAG-augmented microservice retrieving Hindu scripture context from Qdrant Cloud for Claude LLM chat",
    version="0.1.0",
    lifespan=app_lifespan,
)

# Enable CORS using parsed ALLOWED_ORIGINS from settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include endpoint routers
app.include_router(health_router)
app.include_router(chat_router)


@app.get("/")
def root_info():
    """Root endpoint returning service status and information."""
    return {
        "service": "FastAPI Chatbot RAG Microservice",
        "status": "running",
        "version": "0.1.0",
    }