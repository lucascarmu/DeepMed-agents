"""
API route definitions.

Defines the /triage endpoint for multi-turn anamnesis chat.
Handles both initial and follow-up invocations via conversation_id threading.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from app.graph.builder import build_triage_graph
from app.services.database import get_checkpointer

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class TriageRequest(BaseModel):
    """
    Request body for the triage endpoint.

    First call: provide all patient context fields + initial message.
    Follow-up calls: provide only conversation_id + message.
    """

    conversation_id: str = Field(
        ...,
        min_length=1,
        description="Unique conversation/session identifier (used as thread_id)",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Patient's message text",
    )

    # Patient context — required on first call, ignored on follow-ups
    full_name: Optional[str] = Field(
        None, description="Patient's full name (required on first call)"
    )
    age: Optional[int] = Field(
        None, description="Patient's age (required on first call)", ge=0, le=150
    )
    gender: Optional[str] = Field(
        None, description="Patient's gender (required on first call)"
    )
    base_pathologies: Optional[List[str]] = Field(
        default=None,
        description="Known base pathologies/conditions",
    )
    allergies: Optional[List[str]] = Field(
        default=None,
        description="Known allergies",
    )


class TriageResponse(BaseModel):
    """Response from the triage endpoint."""

    conversation_id: str
    assistant_message: str
    anamnesis_complete: bool = False

    # Only populated when anamnesis is complete
    clinical_summary: Optional[Dict[str, Any]] = None
    specialty: Optional[str] = None
    urgency: Optional[str] = None
    routing_result: Optional[str] = None

    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph singleton (compiled once with checkpointer)
# ---------------------------------------------------------------------------

_graph = None


def _get_graph():
    """Lazy-init the graph with PostgreSQL checkpointer."""
    global _graph
    if _graph is None:
        checkpointer = get_checkpointer()
        _graph = build_triage_graph(checkpointer=checkpointer)
    return _graph


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/triage",
    response_model=TriageResponse,
    summary="Multi-turn clinical anamnesis chat",
    description=(
        "Conducts a directed medical anamnesis via multi-turn chat. "
        "On first call, provide patient context. On follow-ups, provide "
        "only conversation_id and message. The system automatically resumes "
        "the previous conversation state."
    ),
)
async def triage_endpoint(request: TriageRequest) -> TriageResponse:
    """
    POST /triage

    Multi-turn anamnesis endpoint. Uses conversation_id as thread_id
    for automatic state persistence and recovery.
    """
    logger.info(
        "POST /triage — conversation_id=%s, message=%d chars",
        request.conversation_id,
        len(request.message),
    )

    try:
        graph = _get_graph()
        result = await _run_graph(graph, request)
        return _build_response(request.conversation_id, result)

    except Exception as exc:
        logger.exception("Triage pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Triage pipeline error: {exc}",
        )


async def _run_graph(graph, request: TriageRequest) -> dict:
    """Execute the graph with the given request, using thread_id for persistence."""
    import asyncio

    config = {
        "configurable": {
            "thread_id": request.conversation_id,
        }
    }

    # Build the input state — always append the new user message
    input_state: dict = {
        "messages": [HumanMessage(content=request.message)],
    }

    # On first invocation, include patient context.
    # On follow-ups, these will be None and we rely on the checkpointer
    # to restore the existing state (patient context already in state).
    if request.full_name is not None:
        input_state["full_name"] = request.full_name
    if request.age is not None:
        input_state["age"] = request.age
    if request.gender is not None:
        input_state["gender"] = request.gender
    if request.base_pathologies is not None:
        input_state["base_pathologies"] = request.base_pathologies
    if request.allergies is not None:
        input_state["allergies"] = request.allergies

    # Set defaults for control fields on every invocation
    input_state["errors"] = []

    # Run synchronous graph.invoke in a thread pool to not block the event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: graph.invoke(input_state, config=config),
    )
    return result


def _build_response(conversation_id: str, result: dict) -> TriageResponse:
    """Build the API response from the graph execution result."""
    return TriageResponse(
        conversation_id=conversation_id,
        assistant_message=result.get("assistant_response", ""),
        anamnesis_complete=result.get("anamnesis_complete", False),
        clinical_summary=result.get("clinical_summary"),
        specialty=result.get("specialty"),
        urgency=result.get("urgency"),
        routing_result=result.get("routing_result"),
        errors=result.get("errors", []),
    )
