"""
Application entry point.

Configures logging, creates the FastAPI application with lifespan for
database connection pool management, and mounts routes.

Run with: uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.database import init_pool, close_pool

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — manages database pool lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the application."""
    logger.info("Starting up — initialising database pool")
    init_pool()
    yield
    logger.info("Shutting down — closing database pool")
    close_pool()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DeepMed Triage API",
    description=(
        "Multi-agent clinical triage system with stateful anamnesis. "
        "Conducts directed medical interviews via multi-turn chat, "
        "then produces structured clinical summaries with specialty "
        "and urgency routing. State is persisted in PostgreSQL."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router, prefix="/api/v1", tags=["triage"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "deepmed-triage", "version": "2.0.0"}


logger.info("DeepMed Triage API initialised (v2.0.0 — stateful anamnesis)")
