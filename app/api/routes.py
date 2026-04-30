"""
API route definitions.

Defines the /triage endpoint and request/response models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.graph.builder import build_triage_graph

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class TriageRequest(BaseModel):
    """Request body for the triage endpoint."""

    input_text: str = Field(
        ...,
        min_length=10,
        description="Raw patient conversation text",
        examples=[
            "I've been having chest pain for 2 days. It gets worse when I breathe deeply. "
            "I also feel short of breath. I have a history of asthma and take albuterol."
        ],
    )


class TriageResponse(BaseModel):
    """Full PatientState returned after graph execution."""

    raw_conversation: str
    enriched_conversation: Optional[str] = None
    clinical_summary: Optional[Dict[str, Any]] = None
    specialty: Optional[str] = None
    urgency: Optional[str] = None
    routing_result: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Compile graph once at module level for reuse
# ---------------------------------------------------------------------------

_graph = None


def _get_graph():
    """Lazy-init graph (deferred so settings are validated at import time)."""
    global _graph
    if _graph is None:
        _graph = build_triage_graph()
    return _graph


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/triage",
    response_model=TriageResponse,
    summary="Process patient conversation through the triage pipeline",
    description=(
        "Accepts raw patient conversation text and runs it through the "
        "multi-agent triage pipeline: triage → structuring → classification → routing."
    ),
)
async def triage_endpoint(request: TriageRequest) -> TriageResponse:
    """
    POST /triage

    Runs the full triage graph and returns the final PatientState.
    """
    logger.info("POST /triage — received %d chars", len(request.input_text))

    try:
        graph = _get_graph()

        # LangGraph invoke is synchronous; run in executor for async compat
        # (LangGraph also supports ainvoke if configured with async nodes)
        result = await _run_graph(graph, request.input_text)

        return TriageResponse(**result)

    except Exception as exc:
        logger.exception("Triage pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Triage pipeline error: {exc}",
        )


async def _run_graph(graph, input_text: str) -> dict:
    """Execute the graph with the given input text."""
    import asyncio

    initial_state = {
        "raw_conversation": input_text,
        "enriched_conversation": None,
        "clinical_summary": None,
        "specialty": None,
        "urgency": None,
        "routing_result": None,
        "errors": [],
    }

    # Run synchronous graph.invoke in a thread pool to not block the event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, graph.invoke, initial_state)
    return result
