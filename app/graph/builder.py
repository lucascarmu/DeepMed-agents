"""
Graph builder.

Constructs the LangGraph StateGraph following langgraph-fundamentals best practices:
  - Explicit node registration
  - Explicit edge wiring (static + conditional)
  - Clean separation between builder and node logic
  - RetryPolicy on LLM-calling nodes
  - compile() with checkpointer for session persistence

Flow (evaluation-first):
    START → evaluation ──(first turn / incomplete)──→ triage → END (wait for user)
                       └─(complete)─────────────────→ farewell → structuring → classifier ──→ routing → END

This order ensures the patient never sees a dangling question when the
anamnesis is already complete. On the turn where completeness is reached,
the farewell node sends a closure message instead of triage asking more.
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from app.schemas.state import PatientState
from app.agents.triage import triage_node
from app.agents.evaluation import evaluation_node
from app.agents.structuring import structuring_node
from app.agents.classifier import classifier_node
from app.graph.nodes import (
    farewell_node,
    emergency_node,
    normal_node,
    route_after_evaluation,
    route_by_urgency,
)

logger = logging.getLogger(__name__)

# Retry policy for LLM-calling nodes (transient network / rate-limit errors)
_LLM_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=1.0)


def build_triage_graph(checkpointer=None):
    """
    Build and compile the medical triage graph.

    Parameters
    ----------
    checkpointer : optional
        A LangGraph checkpointer (e.g. PostgresSaver) for session persistence.
        If None, the graph runs without persistence (useful for testing).

    Returns a compiled graph ready for .invoke() or .stream().
    """
    builder = StateGraph(PatientState)

    # ── Register nodes ──────────────────────────────────────────────────
    builder.add_node("evaluation", evaluation_node, retry_policy=_LLM_RETRY_POLICY)
    builder.add_node("triage", triage_node, retry_policy=_LLM_RETRY_POLICY)
    builder.add_node("farewell", farewell_node)
    builder.add_node("structuring", structuring_node, retry_policy=_LLM_RETRY_POLICY)
    builder.add_node("classifier", classifier_node, retry_policy=_LLM_RETRY_POLICY)
    builder.add_node("emergency_node", emergency_node)
    builder.add_node("normal_node", normal_node)

    # ── Entry edge — evaluate FIRST ─────────────────────────────────────
    builder.add_edge(START, "evaluation")

    # ── Conditional edge after evaluation ───────────────────────────────
    # First turn / incomplete → triage (ask question) → END
    # Complete → farewell (closure message) → structuring pipeline
    builder.add_conditional_edges(
        "evaluation",
        route_after_evaluation,
        ["triage", "farewell"],
    )

    # ── Triage → END (return control to user, wait for next message) ────
    builder.add_edge("triage", END)

    # ── Farewell → Structuring (always) ─────────────────────────────────
    builder.add_edge("farewell", "structuring")

    # ── Structuring → Classifier (always) ───────────────────────────────
    builder.add_edge("structuring", "classifier")

    # ── Conditional edge after classifier ───────────────────────────────
    builder.add_conditional_edges(
        "classifier",
        route_by_urgency,
        ["emergency_node", "normal_node"],
    )

    # ── Terminal edges ──────────────────────────────────────────────────
    builder.add_edge("emergency_node", END)
    builder.add_edge("normal_node", END)

    # ── Compile with checkpointer ───────────────────────────────────────
    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Triage graph compiled successfully (checkpointer=%s)", type(checkpointer).__name__)
    return graph
