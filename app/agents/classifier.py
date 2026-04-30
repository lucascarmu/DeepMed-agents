"""
Classifier Agent.

Receives the structured ClinicalSummary and determines:
  - medical_specialty (string)
  - urgency_level ("low" | "medium" | "high")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.state import PatientState
from app.services.llm import get_llm

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "classifier_prompt.txt"

_VALID_URGENCY = {"low", "medium", "high"}


def _load_prompt() -> str:
    """Load the classifier prompt template from disk."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _parse_json_response(raw_text: str) -> dict:
    """Parse JSON, stripping markdown fences if present."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def classifier_node(state: PatientState) -> dict:
    """
    Classifier node — determines specialty and urgency.

    Input: state['clinical_summary']
    Output: partial state update with 'specialty' and 'urgency'
    """
    clinical_summary = state.get("clinical_summary")

    if clinical_summary is None:
        logger.warning("Classifier node: no clinical summary available, skipping")
        return {
            "specialty": None,
            "urgency": None,
            "errors": ["classifier_node: no clinical_summary in state"],
        }

    summary_json = json.dumps(clinical_summary, indent=2, default=str)
    logger.info("Classifier node: classifying summary (%d chars)", len(summary_json))

    prompt_template = _load_prompt()
    prompt_text = prompt_template.format(clinical_summary=summary_json)

    llm = get_llm()

    messages = [
        SystemMessage(
            content="You are a clinical classification engine. Output ONLY valid JSON."
        ),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = llm.invoke(messages)
        result = _parse_json_response(response.content)

        specialty = result.get("medical_specialty", "unknown")
        urgency = result.get("urgency_level", "medium")

        # Validate urgency value
        if urgency not in _VALID_URGENCY:
            logger.warning(
                "Classifier node: invalid urgency '%s', defaulting to 'medium'",
                urgency,
            )
            urgency = "medium"

        logger.info(
            "Classifier node: specialty=%s, urgency=%s", specialty, urgency
        )
        return {"specialty": specialty, "urgency": urgency}

    except json.JSONDecodeError as exc:
        logger.error("Classifier node: JSON parse error — %s", exc)
        return {
            "specialty": None,
            "urgency": None,
            "errors": [f"classifier_node: JSON parse error — {exc}"],
        }
    except Exception as exc:
        logger.error("Classifier node: unexpected error — %s", exc)
        return {
            "specialty": None,
            "urgency": None,
            "errors": [f"classifier_node: {exc}"],
        }
