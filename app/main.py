"""
Application entry point.

Configures logging, creates the FastAPI application, and mounts routes.
Run with: uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

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
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DeepMed Triage API",
    description=(
        "Multi-agent clinical triage system. "
        "Processes patient conversations through triage, structuring, "
        "and classification agents to produce structured clinical summaries "
        "with specialty and urgency routing."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
    return {"status": "healthy", "service": "deepmed-triage"}


logger.info("DeepMed Triage API initialised")
