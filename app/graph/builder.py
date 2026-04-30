"""
Graph builder.

Constructs the LangGraph StateGraph following langgraph-fundamentals best practices:
  - Explicit node registration
  - Explicit edge wiring (static + conditional)
  - Clean separation between builder and node logic
  - RetryPolicy on LLM-calling nodes
  - compile() before execution
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from app.schemas.state import PatientState
from app.agents.triage import triage_node
from app.agents.structuring import structuring_node
from app.agents.classifier import classifier_node
from app.graph.nodes import emergency_node, normal_node, route_by_urgency

logger = logging.getLogger(__name__)

# Retry policy for LLM-calling nodes (transient network / rate-limit errors)
_LLM_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=1.0)


def build_triage_graph() -> StateGraph:
    """
    Build and compile the medical triage graph.

    Flow:
        START → triage → structuring → classifier ─┐
                                                    ├─ (high)   → emergency_node → END
                                                    └─ (low/med)→ normal_node    → END

    Returns a compiled graph ready for .invoke() or .stream().
    """
    builder = StateGraph(PatientState)

    # ── Register nodes ──────────────────────────────────────────────────
    builder.add_node("triage", triage_node, retry_policy=_LLM_RETRY_POLICY)
    builder.add_node("structuring", structuring_node, retry_policy=_LLM_RETRY_POLICY)
    builder.add_node("classifier", classifier_node, retry_policy=_LLM_RETRY_POLICY)
    builder.add_node("emergency_node", emergency_node)
    builder.add_node("normal_node", normal_node)

    # ── Static edges (deterministic flow) ───────────────────────────────
    builder.add_edge(START, "triage")
    builder.add_edge("triage", "structuring")
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

    # ── Compile ─────────────────────────────────────────────────────────
    graph = builder.compile()
    logger.info("Triage graph compiled successfully")
    return graph
