"""Intent classification + disambiguation (Module 4: Disambiguation).

One Gemini call per turn, response_schema=IntentClassification. Low-confidence
or ambiguous results carry their own clarifying_question — callers should
surface that directly to the employee instead of calling any tool.
"""

from src import config
from src.agent import prompts, usage
from src.schemas import Intent, IntentClassification

def _default_clarifying_question() -> str:
    """Config-driven clarifier: mention the audience segments when the deployment
    defines them, otherwise just ask for more detail. Kept as a function so it
    reflects config at call time (and stays easy to test)."""
    if config.AUDIENCE_CLASSES:
        labels = ", ".join(config.AUDIENCE_LABELS[s] for s in config.AUDIENCE_ORDER)
        return (
            f"Could you tell me a bit more? For example, which {config.AUDIENCE_NOUN} you "
            f"belong to ({labels}), or which topic you mean."
        )
    return "Could you tell me a bit more about what you need help with?"


DEFAULT_CLARIFYING_QUESTION = _default_clarifying_question()
FALLBACK_CLARIFYING_QUESTION = (
    "Could you rephrase that? I want to make sure I route this correctly."
)


def classify_intent(
    message: str,
    history: list[dict[str, str]] | None = None,
    client=None,
    session_id: str | None = None,
) -> IntentClassification:
    from google.genai import types

    client = client or config.get_llm_client()
    response = client.models.generate_content(
        model=config.ACTIVE_CHAT_MODEL,
        contents=prompts.ROUTER_PROMPT.format(
            history=format_history(history or []), message=message
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IntentClassification,
            temperature=0.0,
            thinking_config=config.thinking_config(),
        ),
    )
    usage.record_usage(config.ACTIVE_CHAT_MODEL, usage.extract_usage(response), session_id=session_id)
    result: IntentClassification | None = response.parsed
    if result is None:  # fail closed: ask rather than guess
        return IntentClassification(
            intent=Intent.AMBIGUOUS,
            confidence=0.0,
            clarifying_question=FALLBACK_CLARIFYING_QUESTION,
        )
    if result.confidence < config.ROUTER_CONFIDENCE_FLOOR and not result.clarifying_question:
        result = result.model_copy(
            update={
                "intent": Intent.AMBIGUOUS,
                "clarifying_question": DEFAULT_CLARIFYING_QUESTION,
            }
        )
    return result


def needs_clarification(classification: IntentClassification) -> bool:
    return (
        classification.intent == Intent.AMBIGUOUS
        or classification.confidence < config.ROUTER_CONFIDENCE_FLOOR
    )


def format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
