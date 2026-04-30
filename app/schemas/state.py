"""
LangGraph state definition.

Follows langgraph-fundamentals best practices:
- TypedDict for state schema
- Annotated reducers where list accumulation is needed
- Clean separation from node logic
"""

from __future__ import annotations

from typing import Annotated, Optional

import operator
from typing_extensions import TypedDict

from app.schemas.clinical_summary import ClinicalSummary


class PatientState(TypedDict):
    """
    Shared state that flows through the triage graph.

    Fields use default-overwrite semantics (no reducer) because each field
    is written by exactly one node and should not accumulate:
      - raw_conversation: set at entry
      - enriched_conversation: set by triage node
      - clinical_summary: set by structuring node
      - specialty: set by classifier node
      - urgency: set by classifier node
      - routing_result: set by post-classifier routing node
      - errors: accumulates via reducer — any node may append errors
    """

    raw_conversation: str
    enriched_conversation: Optional[str]
    clinical_summary: Optional[dict]  # serialised ClinicalSummary
    specialty: Optional[str]
    urgency: Optional[str]
    routing_result: Optional[str]
    errors: Annotated[list[str], operator.add]
