"""
Triage Agent — Directed Anamnesis.

Conducts a multi-turn clinical interview (anamnesis) with the patient.
Reads the full chat history + patient context from state, generates the
next question or response, and appends it to the messages list.

If evaluation_reasoning is available from a prior turn, it's injected
into the prompt so the agent knows what information is still missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.schemas.state import PatientState
from app.services.llm import get_llm

logger = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "triage_anamnesis_prompt.txt"
)


def _load_prompt() -> str:
    """Load the anamnesis system prompt template from disk."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _format_list(items: list[str]) -> str:
    """Format a list for display in the prompt, handling empty lists."""
    if not items:
        return "Ninguna informada"
    return ", ".join(items)


def _build_system_prompt(state: PatientState) -> str:
    """
    Build the system prompt with patient context injected.

    If evaluation_reasoning is present from a prior evaluation cycle,
    appends it as guidance so the triage knows what info is still missing.
    """
    template = _load_prompt()
    prompt = template.format(
        full_name=state["full_name"],
        age=state["age"],
        gender=state["gender"],
        base_pathologies=_format_list(state.get("base_pathologies", [])),
        allergies=_format_list(state.get("allergies", [])),
    )

    # Inject evaluation reasoning if available
    reasoning = state.get("evaluation_reasoning")
    if reasoning:
        prompt += (
            "\n\n--- CONTEXTO DEL EVALUADOR CLÍNICO ---\n"
            "El evaluador de anamnesis determinó que aún falta información. "
            "Su análisis:\n"
            f'"{reasoning}"\n'
            "Usa esta guía para dirigir tus próximas preguntas hacia lo que falta.\n"
            "--- FIN DEL CONTEXTO ---"
        )

    return prompt


def _state_messages_to_langchain(messages: list) -> list:
    """
    Convert state message dicts back to LangChain message objects.

    Messages in state may be dicts (from serialisation) or already
    LangChain message objects.
    """
    result = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
            result.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("type") or msg.get("role", "human")
            content = msg.get("content", "")
            if role in ("human", "user"):
                result.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                result.append(AIMessage(content=content))
            elif role == "system":
                result.append(SystemMessage(content=content))
        # skip anything else
    return result


def triage_node(state: PatientState) -> dict:
    """
    Triage node — directed anamnesis conversational agent.

    Reads the full message history and patient context, generates the
    next response in the clinical interview.

    Input: state['messages'], patient context fields, evaluation_reasoning
    Output: partial state update appending AI response to messages
    """
    messages = state.get("messages", [])
    logger.info(
        "Triage node: processing conversation with %d messages", len(messages)
    )

    system_prompt = _build_system_prompt(state)
    llm = get_llm()

    # Build the full message list for the LLM
    lc_messages = [SystemMessage(content=system_prompt)]
    lc_messages.extend(_state_messages_to_langchain(messages))

    try:
        response = llm.invoke(lc_messages)
        assistant_text = response.content.strip()
        logger.info(
            "Triage node: generated response (%d chars)", len(assistant_text)
        )

        # Return partial update: append AI message + set assistant_response
        return {
            "messages": [AIMessage(content=assistant_text)],
            "assistant_response": assistant_text,
        }
    except Exception as exc:
        logger.error("Triage node failed: %s", exc)
        fallback = (
            "Disculpe, ocurrió un error temporal. "
            "¿Podría repetir su último mensaje?"
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "assistant_response": fallback,
            "errors": [f"triage_node: {exc}"],
        }
