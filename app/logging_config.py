"""
Centralized logging setup for the service (API + CLI scripts).

Everything lives under the "app" logger namespace so API routes, services,
and standalone scripts (e.g. app/scripts/ingest.py) share one consistent,
timestamped log format independent of uvicorn's own logging setup. A
per-request ID is threaded through via a ContextVar so concurrent requests
can be told apart in the logs.
"""

import logging
import sys
from contextvars import ContextVar
from urllib.parse import urlsplit, urlunsplit

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | req=%(request_id)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """
    Configures the "app" logger (parent of every app.* module logger) with a
    single stdout stream handler. Idempotent/safe to call more than once
    (e.g. under `uvicorn --reload`).
    """
    global _configured

    app_logger = logging.getLogger("app")
    app_logger.setLevel(level.upper())
    # Don't propagate to the root logger to avoid double-logging alongside
    # uvicorn's own "uvicorn.*" loggers.
    app_logger.propagate = False

    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        handler.addFilter(_RequestIdFilter())
        app_logger.addHandler(handler)
        _configured = True

    # Quiet down chatty third-party HTTP clients; their request/response logs
    # can incidentally include upstream URLs and aren't useful at INFO here.
    for noisy_logger in ("httpx", "httpcore", "hpack", "qdrant_client"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def redact_url(url: str) -> str:
    """
    Strips credentials and query parameters from a connection URL so it is
    safe to log. Keeps only scheme, host, port, and path.

    e.g. "mongodb://user:pass@host:27017/db?authSource=admin"
         -> "mongodb://host:27017/db"
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        netloc = parts.hostname or ""
        if parts.port:
            netloc += f":{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except Exception:
        return "<redacted>"
