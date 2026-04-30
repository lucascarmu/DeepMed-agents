"""
Graph node wrappers and routing logic.

Separates post-processing nodes (emergency, normal) and routing functions
from the core agent logic. This keeps the graph builder clean.
"""

from __future__ import annotations

import logging
from typing import Literal

from app.schemas.state import PatientState

logger = logging.getLogger(__name__)


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
# Conditional routing function
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
