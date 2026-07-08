"""FastAPI interface for the HR assistant.

Member 4 owns this boundary: HTTP contracts, request validation, sanitized
monitoring, and API-shaped responses for the Streamlit UI. When Member 2's
agent orchestrator is present it can be plugged in without changing clients;
until then the endpoint serves grounded FAQ answers through the RAG module.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.monitoring import chat_trace, configure_mlflow
from src.rag.answerer import answer_question
from src.rag.retriever import RetrievedChunk
from src.schemas import Citation, TokenUsage


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
        description="Optional retrieval filter: leave|benefits|payroll|conduct|complaints|onboarding",
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
    ticket_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourceResponse] = Field(default_factory=list)
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
        "Covers the agent's own Gemini calls only (router, ReAct loop, search_web). "
        "Does not include the plain-RAG fallback path or ingestion embedding calls."
    )


app = FastAPI(
    title="HR FAQ & Complaint Chatbot API",
    version="0.1.0",
    description="REST API for grounded HR policy answers and complaint workflow actions.",
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


def _complaint_intake_pending(message: str) -> bool:
    lowered = message.lower()
    complaint_terms = ("complaint", "report", "harassment", "discrimination", "unsafe", "grievance")
    return any(term in lowered for term in complaint_terms)


# In-process, non-persistent turn history keyed by session_id. Stands in for
# the real session/long-term memory store (src/memory/) so multi-turn
# conversations (e.g. complaint slot-filling) work across requests before
# that module exists; history is lost on server restart.
_SESSION_HISTORY: dict[str, list[dict[str, str]]] = {}


def _try_agent_orchestrator(request: ChatRequest, session_id: str) -> ChatResponse | None:
    """Use Member 2's orchestrator when it exists.

    Supported future shape: handle_message(message=..., session_id=...,
    employee_id=..., history=...) returning either ChatResponse, dict, or
    object with response-like attributes.
    """
    try:
        from src.agent.orchestrator import handle_message
    except Exception:
        return None

    history = _SESSION_HISTORY.get(session_id, [])
    result = handle_message(
        message=request.message,
        session_id=session_id,
        employee_id=request.employee_id,
        history=history,
    )
    if isinstance(result, ChatResponse):
        response = result
    elif isinstance(result, dict):
        response = ChatResponse.model_validate(result)
    else:
        response = ChatResponse.model_validate(result.model_dump())

    _SESSION_HISTORY[session_id] = history + [
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": response.reply},
    ]
    return response


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid4())
    with chat_trace(session_id=session_id, message=request.message) as trace:
        agent_response = _try_agent_orchestrator(request, session_id)
        if agent_response is not None:
            trace["metrics"] = {
                "citation_count": len(agent_response.citations),
                "source_count": len(agent_response.sources),
                "action_count": len(agent_response.actions),
                "prompt_tokens": agent_response.token_usage.prompt_tokens,
                "completion_tokens": agent_response.token_usage.completion_tokens,
                "total_tokens": agent_response.token_usage.total_tokens,
            }
            trace["tags"] = {"route": "agent", "insufficient_context": agent_response.insufficient_context}
            return agent_response

        answer, chunks = answer_question(request.message, category=request.category)
        actions: list[ActionResponse] = []
        if _complaint_intake_pending(request.message):
            actions.append(
                ActionResponse(
                    type="complaint_intake",
                    label="Complaint intake requires the agent/tool-use module.",
                    status="pending",
                )
            )

        response = ChatResponse(
            session_id=session_id,
            reply=answer.answer,
            citations=answer.citations,
            sources=[_source_from_chunk(chunk) for chunk in chunks],
            actions=actions,
            insufficient_context=answer.insufficient_context,
        )
        trace["metrics"] = {
            "citation_count": len(response.citations),
            "source_count": len(response.sources),
            "action_count": len(response.actions),
        }
        trace["tags"] = {"route": "rag", "insufficient_context": response.insufficient_context}
        return response


def _fetch_ticket(ticket_id: str, db_path: Path = config.SQLITE_PATH) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('tickets', 'complaint_tickets')"
        ).fetchall()
        for row in table_rows:
            table = row["name"]
            columns = {
                column["name"]
                for column in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            id_columns = [column for column in ("ticket_id", "id") if column in columns]
            for id_column in id_columns:
                result = conn.execute(
                    f"SELECT * FROM {table} WHERE {id_column} = ?", (ticket_id,)
                ).fetchone()
                if result:
                    return dict(result)
    return None


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str) -> dict[str, Any]:
    ticket = _fetch_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


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
