"""Tools the answer pipeline calls (Module 8: Tool Use).

The orchestrator's pipeline (Phase 5) calls these directly in a fixed order —
search_kb first, then search_web only if the KB is insufficient — rather than a
model choosing between them in a ReAct loop.

- search_kb    -> internal RAG, delegates to answer_question() in src/rag/answerer.py
- search_web   -> fallback for national statutory pre-employment questions (NBI, SSS,
                  PhilHealth, Pag-IBIG, BIR) the internal KB doesn't cover; Tavily search
                  (Philippines-grounded, domain-restricted, provider-agnostic) + one
                  structured Gemini call to shape results into a GroundedAnswer
"""

import logging

from src import config
from src.agent import prompts, usage
from src.rag.answerer import answer_question
from src.rag.retriever import RetrievedChunk
from src.schemas import AnswerSource, GroundedAnswer, WebCitation

logger = logging.getLogger(__name__)

NO_WEB_ANSWER = (
    "I couldn't find a reliable official government source for this either, so I don't "
    f"want to guess. I can route your question to {config.HELP_CONTACT} instead — "
    "would you like that?"
)


# --- search_kb (internal RAG) -------------------------------------------------

def search_kb(
    question: str, category: str | None = None, client=None, session_id: str | None = None
) -> tuple[GroundedAnswer, list[RetrievedChunk]]:
    """Internal knowledge-base RAG tool. Thin wrapper so the orchestrator has a
    single tool-call surface; all retrieval/grounding logic lives in
    src/rag/answerer.py. session_id is threaded so the grounded-answer call is
    accounted under the turn."""
    return answer_question(question, category=category, client=client, session_id=session_id)


# --- search_web (statutory pre-employment fallback) ---------------------------

def search_web(
    question: str, client=None, session_id: str | None = None, tavily_client=None
) -> GroundedAnswer:
    """Fallback for questions the internal KB doesn't cover (e.g. national statutory
    pre-employment requirements from NBI/SSS/PhilHealth/Pag-IBIG/BIR).

    Tavily does the actual searching (domain-restricted, works the same regardless
    of which LLM serves chat), then one Gemini call with response_schema shapes the
    results into a typed GroundedAnswer.
    """
    from google.genai import types

    results = _tavily_search(question, tavily_client=tavily_client)
    if not results:
        return no_web_answer()

    client = client or config.get_llm_client()
    shape_response = client.models.generate_content(
        model=config.ACTIVE_CHAT_MODEL,
        contents=prompts.WEB_ANSWER_SHAPE_PROMPT.format(
            question=question, search_results=_format_tavily_results(results)
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GroundedAnswer,
            temperature=0.0,
        ),
    )
    usage.record_usage(config.ACTIVE_CHAT_MODEL, usage.extract_usage(shape_response), session_id=session_id)
    answer: GroundedAnswer | None = shape_response.parsed
    if answer is None or answer.insufficient_context:  # fail closed
        return no_web_answer()

    web_citations = [
        WebCitation(url=r["url"], title=r.get("title") or r["url"], snippet=(r.get("content") or "")[:300])
        for r in results
    ]
    return answer.model_copy(
        update={"source": AnswerSource.WEB, "web_citations": web_citations, "citations": []}
    )


def _tavily_search(question: str, tavily_client=None) -> list[dict]:
    """Country- and domain-restricted Tavily search. Grounded in the Philippines
    two ways: `country` (config.SEARCH_COUNTRY) boosts PH sources, and
    `include_domains` (config.STATUTORY_GOV_DOMAINS) hard-restricts to the official
    PH government agencies the fallback covers (NBI/SSS/PhilHealth/Pag-IBIG/BIR/
    DOLE). Fails closed (empty list) on API errors rather than raising — a search
    outage shouldn't crash the whole agent turn. `country` is passed as a kwarg so
    older Tavily SDKs that don't accept it degrade to domain-only grounding."""
    from tavily.errors import (
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        UsageLimitExceededError,
    )
    from tavily.errors import TimeoutError as TavilyTimeoutError

    tavily_client = tavily_client or config.get_tavily_client()
    search_kwargs = dict(
        query=question,
        include_domains=list(config.STATUTORY_GOV_DOMAINS),
        max_results=config.TAVILY_MAX_RESULTS,
    )
    if config.SEARCH_COUNTRY:
        search_kwargs["country"] = config.SEARCH_COUNTRY
    try:
        response = tavily_client.search(**search_kwargs)
    except TypeError:
        # SDK too old for the `country` kwarg — retry with domain grounding only.
        search_kwargs.pop("country", None)
        response = tavily_client.search(**search_kwargs)
    except (
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        UsageLimitExceededError,
        TavilyTimeoutError,
    ) as exc:
        logger.warning("tavily_search_error question=%r error=%s", question, exc)
        return []
    return response.get("results") or []


def _format_tavily_results(results: list[dict]) -> str:
    parts = []
    for r in results:
        parts.append(f"[url: {r['url']} | title: {r.get('title', '')}]\n{r.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def no_web_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer=NO_WEB_ANSWER,
        citations=[],
        source=AnswerSource.NONE,
        web_citations=[],
        insufficient_context=True,
    )

