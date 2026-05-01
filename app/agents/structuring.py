"""
Structuring Agent.

Takes the full multi-turn conversation from state['messages'] and produces
a strict ClinicalSummary JSON object validated against the Pydantic model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.schemas.clinical_summary import ClinicalSummary
from app.schemas.state import PatientState
from app.services.llm import get_llm

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "structuring_prompt.txt"


def _load_prompt() -> str:
    """Load the structuring prompt template from disk."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _messages_to_transcript(messages: list) -> str:
    """Convert message list to a readable transcript for the LLM."""
    lines = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("type") or msg.get("role", "unknown")
            content = msg.get("content", "")
        elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
            role = msg.type
            content = msg.content
        else:
            continue

        if role in ("human", "user"):
            lines.append(f"PATIENT: {content}")
        elif role in ("ai", "assistant"):
            lines.append(f"ASSISTANT: {content}")
    return "\n".join(lines)


def _parse_json_response(raw_text: str) -> dict:
    """
    Attempt to parse JSON from the LLM response.

    Handles cases where the model wraps output in markdown code fences.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last line (``` markers)
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return json.loads(text)


def structuring_node(state: PatientState) -> dict:
    """
    Structuring node — converts the full conversation to ClinicalSummary.

    Input: state['messages'] (full multi-turn conversation)
    Output: partial state update with 'clinical_summary' (serialised dict)
    """
    messages = state.get("messages", [])
    conversation = _messages_to_transcript(messages)
    logger.info("Structuring node: processing conversation (%d chars)", len(conversation))

    prompt_template = _load_prompt()
    prompt_text = prompt_template.format(conversation=conversation)

    llm = get_llm()

    lc_messages = [
        SystemMessage(
            content="You are a clinical data extraction engine. Output ONLY valid JSON."
        ),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = llm.invoke(lc_messages)
        raw_json = _parse_json_response(response.content)

        # Validate through Pydantic — this enforces the strict schema
        summary = ClinicalSummary(**raw_json)
        logger.info("Structuring node: ClinicalSummary validated successfully")

        return {"clinical_summary": summary.model_dump(mode="json")}

    except json.JSONDecodeError as exc:
        logger.error("Structuring node: JSON parse error — %s", exc)
        return {
            "clinical_summary": None,
            "errors": [f"structuring_node: JSON parse error — {exc}"],
        }
    except Exception as exc:
        logger.error("Structuring node: unexpected error — %s", exc)
        return {
            "clinical_summary": None,
            "errors": [f"structuring_node: {exc}"],
        }
