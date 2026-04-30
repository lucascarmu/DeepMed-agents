"""
Triage Agent.

Receives the raw patient conversation and enriches it by appending
clarification prompts or normalising the text for downstream processing.

This is the first node in the graph pipeline.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.state import PatientState
from app.services.llm import get_llm

logger = logging.getLogger(__name__)

_TRIAGE_SYSTEM_PROMPT = (
    "You are a medical triage assistant. Your job is to review a patient's "
    "raw conversation and produce an enriched version that:\n"
    "1. Preserves ALL original information.\n"
    "2. Normalises medical terminology where appropriate.\n"
    "3. Highlights any missing critical information by appending a short "
    "'[CLARIFICATION NEEDED]' note at the end listing what is missing "
    "(e.g. duration, severity, onset).\n"
    "4. Does NOT add diagnoses or medical opinions.\n"
    "5. Returns ONLY the enriched conversation text — no extra commentary."
)


def triage_node(state: PatientState) -> dict:
    """
    Triage node — enriches the raw patient conversation.

    Input: state['raw_conversation']
    Output: partial state update with 'enriched_conversation'
    """
    raw = state["raw_conversation"]
    logger.info("Triage node: processing %d chars of conversation", len(raw))

    llm = get_llm()

    messages = [
        SystemMessage(content=_TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=raw),
    ]

    try:
        response = llm.invoke(messages)
        enriched = response.content.strip()
        logger.info("Triage node: enriched conversation produced (%d chars)", len(enriched))
        return {"enriched_conversation": enriched}
    except Exception as exc:
        logger.error("Triage node failed: %s", exc)
        return {
            "enriched_conversation": raw,  # fallback to original
            "errors": [f"triage_node: {exc}"],
        }
