import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import db_container
from app.logging_config import configure_logging, redact_url, request_id_var
import motor.motor_asyncio
import redis.asyncio as redis

from app.routes.health import router as health_router
from app.routes.chat import router as chat_router
from app.services.retrieval import get_qdrant_client

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """
    FastAPI Lifespan event handler:
    1. Connects to MongoDB and Redis async clients.
    2. Configures API-based embedding service.
    3. Verifies Qdrant Cloud connection and credentials.
    """
    logger.info("Starting FastAPI Chatbot RAG Microservice...")

    # 1. Connect MongoDB & Redis
    logger.info(
        "Connecting to MongoDB (%s) and Redis (%s)...",
        redact_url(settings.MONGO_URI),
        redact_url(settings.REDIS_URL),
    )
    db_container.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGO_URI)
    db_container.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

    # 2. Embedding service configuration
    logger.info("API embedding service initialized for model '%s'.", settings.EMBEDDING_MODEL_NAME)

    # 3. Verify Qdrant Cloud connection & credentials
    logger.info("Verifying Qdrant Cloud connectivity (%s)...", redact_url(settings.QDRANT_URL))
    try:
        q_client = get_qdrant_client()
        collections_resp = q_client.get_collections()
        existing = [c.name for c in collections_resp.collections]
        logger.info("Qdrant Cloud connection verified. Existing collections: %s", existing)
    except Exception:
        logger.exception("Unable to connect to Qdrant Cloud. Check QDRANT_URL and QDRANT_API_KEY.")
        raise

    logger.info("Microservice startup complete! Ready to accept requests.")

    yield

    # Shutdown: Close connections
    logger.info("Closing database & cache connections...")
    if db_container.mongo_client:
        db_container.mongo_client.close()
    if db_container.redis_client:
        await db_container.redis_client.aclose()
    logger.info("Microservice shutdown complete.")


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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Tags every request with a short correlation ID (exposed via the
    X-Request-ID response header) and logs method/path/status/duration.
    Never logs headers or body, since those can carry the JWT bearer token
    or user-submitted chat content.
    """
    req_id = uuid.uuid4().hex[:8]
    token = request_id_var.set(req_id)
    start = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception("%s %s failed after %.1fms", request.method, request.url.path, duration_ms)
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = req_id
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
    finally:
        request_id_var.reset(token)


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
