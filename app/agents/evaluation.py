"""
Evaluation Agent — Anamnesis Completeness Assessment.

Evaluates whether the clinical interview has gathered enough information
to proceed to the structuring and classification pipeline.

Uses an LLM call with structured JSON output to determine completeness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.schemas.state import PatientState
from app.services.llm import get_llm

logger = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "evaluation_prompt.txt"
)


def _load_prompt() -> str:
    """Load the evaluation prompt template from disk."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _messages_to_transcript(messages: list) -> str:
    """Convert message list to a readable transcript string."""
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
            lines.append(f"PACIENTE: {content}")
        elif role in ("ai", "assistant"):
            lines.append(f"ASSISTENTE: {content}")
    return "\n".join(lines)


def _parse_json_response(raw_text: str) -> dict:
    """Parse JSON, stripping markdown fences if present."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def evaluation_node(state: PatientState) -> dict:
    """
    Evaluation node — determines if anamnesis is complete.

    Reads the full conversation transcript and uses an LLM to assess
    whether sufficient clinical information has been gathered.

    Input: state['messages']
    Output: partial state update with 'anamnesis_complete'
    """
    messages = state.get("messages", [])
    logger.info(
        "Evaluation node: assessing completeness (%d messages)", len(messages)
    )

    # Need at least 2 messages (1 patient + 1 assistant) to evaluate
    # Count human messages only
    human_count = sum(
        1 for m in messages
        if (isinstance(m, dict) and (m.get("type") or m.get("role", "")) in ("human", "user"))
        or (isinstance(m, HumanMessage))
    )

    if human_count < 2:
        logger.info(
            "Evaluation node: only %d patient messages — too early, continuing anamnesis",
            human_count,
        )
        return {"anamnesis_complete": False}

    transcript = _messages_to_transcript(messages)
    prompt_template = _load_prompt()
    prompt_text = prompt_template.format(transcript=transcript)

    llm = get_llm()

    lc_messages = [
        SystemMessage(
            content="You are a clinical evaluation engine. Output ONLY valid JSON."
        ),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = llm.invoke(lc_messages)
        result = _parse_json_response(response.content)

        is_complete = result.get("is_complete", False)
        reasoning = result.get("reasoning", "")

        logger.info(
            "Evaluation node: complete=%s, reasoning=%s",
            is_complete,
            reasoning[:100],
        )
        return {
            "anamnesis_complete": bool(is_complete),
            "evaluation_reasoning": reasoning,
        }

    except json.JSONDecodeError as exc:
        logger.error("Evaluation node: JSON parse error — %s", exc)
        # On parse failure, continue anamnesis (err on side of more data)
        return {
            "anamnesis_complete": False,
            "errors": [f"evaluation_node: JSON parse error — {exc}"],
        }
    except Exception as exc:
        logger.error("Evaluation node: unexpected error — %s", exc)
        return {
            "anamnesis_complete": False,
            "errors": [f"evaluation_node: {exc}"],
        }
