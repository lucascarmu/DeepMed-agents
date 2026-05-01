"""
LangGraph state definition for the stateful anamnesis system.

Follows langgraph-fundamentals best practices:
- TypedDict for state schema
- Annotated reducers for list accumulation (messages, errors)
- Clean separation from node logic

The graph is designed for multi-turn chat: each invocation appends new
messages and the checkpointer persists the full state between turns.
"""

from __future__ import annotations

from typing import Annotated, Optional

import operator
from typing_extensions import TypedDict


class PatientState(TypedDict):
    """
    Shared state that flows through the triage graph.

    Accumulating fields (reducer = operator.add):
      - messages: full chat history (serialised HumanMessage / AIMessage dicts)
      - errors: any node may append error strings

    Overwrite fields (set once or updated by a single node):
      - full_name, age, gender: patient demographics (set at first invocation)
      - base_pathologies, allergies: known medical background
      - anamnesis_complete: set by evaluation node
      - assistant_response: latest bot reply to return via API
      - clinical_summary: set by structuring node
      - specialty, urgency: set by classifier node
      - routing_result: set by post-classifier routing node
    """

    # ── Chat history (append-reducer) ───────────────────────────────────
    messages: Annotated[list, operator.add]

    # ── Patient context (set once at first invocation) ──────────────────
    full_name: str
    age: int
    gender: str
    base_pathologies: list[str]
    allergies: list[str]

    # ── Anamnesis control ───────────────────────────────────────────────
    anamnesis_complete: bool
    evaluation_reasoning: Optional[str]
    assistant_response: Optional[str]

    # ── Downstream pipeline outputs ─────────────────────────────────────
    clinical_summary: Optional[dict]  # serialised ClinicalSummary
    specialty: Optional[str]
    urgency: Optional[str]
    routing_result: Optional[str]

    # ── Error accumulator (append-reducer) ──────────────────────────────
    errors: Annotated[list[str], operator.add]
