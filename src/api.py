"""FastAPI interface for the assistant (branding/scope from src/config.py).

Member 4 owns this boundary: HTTP contracts, request validation, sanitized
monitoring, and API-shaped responses for the Streamlit UI. When Member 2's
agent orchestrator is present it can be plugged in without changing clients;
until then the endpoint serves grounded FAQ answers through the RAG module.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.monitoring import chat_trace, configure_mlflow
from src.rag.answerer import answer_question
from src.rag.retriever import RetrievedChunk
from src.schemas import Citation, TokenUsage, WebCitation


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_mlflow()
    yield


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Client-generated chat session ID")
    message: str = Field(min_length=1, max_length=4000)
    employee_id: str | None = Field(default=None, description="Optional stable employee ID")
    category: str | None = Field(
        default=None,
        description="Optional retrieval filter: onboarding|conduct|leave|benefits",
    )


class SourceResponse(BaseModel):
    chunk_id: str
    title: str
    section_path: str
    similarity: float
    effective_date: str = ""
    version: str = ""
    preview: str


class ActionResponse(BaseModel):
    type: str
    label: str
    status: Literal["completed", "pending", "unavailable"] = "completed"


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourceResponse] = Field(default_factory=list)
    web_citations: list[WebCitation] = Field(default_factory=list)
    actions: list[ActionResponse] = Field(default_factory=list)
    insufficient_context: bool = False
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    chroma_dir_exists: bool
    manifest_exists: bool
    gemini_api_key_configured: bool
    mlflow_tracking_uri: str


class UsageResponse(BaseModel):
    today: dict[str, int]
    all_time: dict[str, int]
    note: str = (
        "Covers the agent's own LLM calls only (router, ReAct loop, search_web), "
        "combined across whichever backend(s) LLM_PROVIDER has pointed at over "
        "time — Gemini and Ollama usage are logged with different model labels in "
        "the underlying token_usage table but summed together here. Does not "
        "include the plain-RAG fallback path or ingestion embedding calls, which "
        "always use Gemini."
    )


app = FastAPI(
    title=f"{config.ASSISTANT_NAME} API",
    version="0.1.0",
    description=f"REST API for grounded answers about {config.SCOPE_PHRASE}.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _source_from_chunk(chunk: RetrievedChunk) -> SourceResponse:
    preview = " ".join(chunk.text.split())
    if len(preview) > 360:
        preview = preview[:357].rstrip() + "..."
    return SourceResponse(
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        section_path=chunk.section_path,
        similarity=round(chunk.similarity, 4),
        effective_date=chunk.effective_date,
        version=chunk.version,
        preview=preview,
    )


def _try_agent_orchestrator(request: ChatRequest, session_id: str) -> ChatResponse | None:
    """Use the agent orchestrator when it exists.

    Supported shape: handle_message(message=..., session_id=..., employee_id=...)
    returning either ChatResponse, dict, or object with response-like
    attributes. `history` is deliberately not passed — leaving it unset tells
    handle_message() to manage session/long-term memory itself via
    src/memory/ (SQLite-backed, survives a restart), rather than api.py
    maintaining its own in-process copy.
    """
    try:
        from src.agent.orchestrator import handle_message
    except Exception:
        return None

    result = handle_message(
        message=request.message,
        session_id=session_id,
        employee_id=request.employee_id,
    )
    if isinstance(result, ChatResponse):
        return result
    if isinstance(result, dict):
        return ChatResponse.model_validate(result)
    return ChatResponse.model_validate(result.model_dump())


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid4())
    with chat_trace(session_id=session_id, message=request.message) as trace:
        agent_response = _try_agent_orchestrator(request, session_id)
        if agent_response is not None:
            trace["metrics"] = {
                "citation_count": len(agent_response.citations),
                "source_count": len(agent_response.sources),
                "web_citation_count": len(agent_response.web_citations),
                "action_count": len(agent_response.actions),
                "prompt_tokens": agent_response.token_usage.prompt_tokens,
                "completion_tokens": agent_response.token_usage.completion_tokens,
                "total_tokens": agent_response.token_usage.total_tokens,
            }
            trace["tags"] = {
                "route": "agent",
                "insufficient_context": agent_response.insufficient_context,
            }
            return agent_response

        answer, chunks = answer_question(request.message, category=request.category)
        response = ChatResponse(
            session_id=session_id,
            reply=answer.answer,
            citations=answer.citations,
            sources=[_source_from_chunk(chunk) for chunk in chunks],
            insufficient_context=answer.insufficient_context,
        )
        trace["metrics"] = {
            "citation_count": len(response.citations),
            "source_count": len(response.sources),
            "action_count": len(response.actions),
        }
        trace["tags"] = {"route": "rag", "insufficient_context": response.insufficient_context}
        return response


@app.get("/usage", response_model=UsageResponse)
def usage() -> UsageResponse:
    try:
        from src.agent import usage as usage_tracker
    except Exception:
        empty = {"request_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return UsageResponse(today=empty, all_time=empty)

    return UsageResponse(
        today=usage_tracker.get_usage_today(),
        all_time=usage_tracker.get_usage_all_time(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    gemini_key = bool(__import__("os").environ.get("GEMINI_API_KEY"))
    chroma_exists = config.CHROMA_DIR.exists()
    manifest_exists = config.MANIFEST_PATH.exists()
    return HealthResponse(
        status="ok" if chroma_exists and manifest_exists and gemini_key else "degraded",
        chroma_dir_exists=chroma_exists,
        manifest_exists=manifest_exists,
        gemini_api_key_configured=gemini_key,
        mlflow_tracking_uri=config.MLFLOW_TRACKING_URI,
    )
