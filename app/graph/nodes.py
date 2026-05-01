"""
Graph node wrappers and routing logic.

Separates routing functions from core agent logic.
Contains:
  - route_after_evaluation: conditional edge after evaluation node
  - farewell_node: closure message when anamnesis is complete
  - route_by_urgency: conditional edge after classifier node
  - emergency_node / normal_node: post-classifier routing nodes
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END

from app.schemas.state import PatientState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Post-evaluation routing
# ---------------------------------------------------------------------------

def route_after_evaluation(
    state: PatientState,
) -> Literal["triage", "farewell", "__end__"]:
    """
    Routing function after the evaluation node.

    - First turn (no AI messages yet) → always go to triage (needs greeting)
    - Anamnesis incomplete → triage (ask next question)
    - Anamnesis complete → farewell (send closure message before pipeline)
    """
    is_complete = state.get("anamnesis_complete", False)
    messages = state.get("messages", [])

    # Check if the bot has spoken at all (first turn detection)
    has_ai_messages = any(
        (isinstance(m, AIMessage))
        or (isinstance(m, dict) and (m.get("type") or m.get("role", "")) in ("ai", "assistant"))
        for m in messages
    )

    if not has_ai_messages:
        logger.info("Routing: first turn — no AI messages yet → triage")
        return "triage"

    if is_complete:
        logger.info("Routing: anamnesis COMPLETE → farewell")
        return "farewell"

    logger.info("Routing: anamnesis INCOMPLETE → triage")
    return "triage"


# ---------------------------------------------------------------------------
# Farewell node — closure message when anamnesis is complete
# ---------------------------------------------------------------------------

def farewell_node(state: PatientState) -> dict:
    """
    Farewell node — sends a closure message to the patient before
    proceeding to the structuring pipeline.

    This ensures the patient always gets a proper goodbye instead of
    seeing a dangling question from the triage agent.
    """
    full_name = state.get("full_name", "")
    first_name = full_name.split()[0] if full_name else ""

    farewell_text = (
        f"Muchas gracias por su tiempo, {first_name}. "
        "Ya tengo toda la información necesaria para procesar su caso. "
        "En unos momentos recibirá el resultado de su evaluación clínica."
    )

    logger.info("Farewell node: sending closure message to patient")
    return {
        "messages": [AIMessage(content=farewell_text)],
        "assistant_response": farewell_text,
    }


# ---------------------------------------------------------------------------
# Post-classifier routing nodes
# ---------------------------------------------------------------------------

def emergency_node(state: PatientState) -> dict:
    """
    Emergency routing node.

    Activated when classifier determines urgency == 'high'.
    In production, this would trigger alerts, escalations, etc.
    """
    logger.warning(
        "EMERGENCY: Patient routed to emergency pathway — specialty=%s",
        state.get("specialty"),
    )
    return {
        "routing_result": "emergency_pathway_activated",
    }


def normal_node(state: PatientState) -> dict:
    """
    Normal routing node.

    Activated when urgency is 'low' or 'medium'.
    In production, this would schedule normal appointments, etc.
    """
    logger.info(
        "Normal pathway: Patient routed normally — specialty=%s, urgency=%s",
        state.get("specialty"),
        state.get("urgency"),
    )
    return {
        "routing_result": "normal_pathway",
    }


# ---------------------------------------------------------------------------
# Post-classifier routing function
# ---------------------------------------------------------------------------

def route_by_urgency(state: PatientState) -> Literal["emergency_node", "normal_node"]:
    """
    Routing function for conditional edges after the classifier node.

    Returns the name of the next node based on urgency level.
    """
    urgency = state.get("urgency")

    if urgency == "high":
        logger.info("Routing decision: HIGH urgency → emergency_node")
        return "emergency_node"

    logger.info("Routing decision: %s urgency → normal_node", urgency)
    return "normal_node"
