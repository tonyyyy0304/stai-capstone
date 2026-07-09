"""Streamlit chat UI for the HR assistant."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.schemas import ComplaintCategory, Severity

_CATEGORY_OPTIONS = [c.value for c in ComplaintCategory]
_SEVERITY_OPTIONS = [s.value for s in Severity]


st.set_page_config(page_title="HR Assistant", page_icon="HR", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi. Ask me about HR policies, or tell me if you need to start a complaint.",
            "citations": [],
            "sources": [],
            "web_citations": [],
            "actions": [],
        }
    ]

st.title("HR Assistant")
st.caption("Grounded HR policy answers with citations. Complaint intake connects through the API workflow.")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API URL", value=config.API_URL, label_visibility="visible")
    if st.button("Check API", use_container_width=True):
        try:
            health = requests.get(f"{api_url.rstrip('/')}/health", timeout=10)
            if health.ok:
                st.success(f"API status: {health.json()['status']}")
            else:
                st.error(f"API returned {health.status_code}")
        except requests.RequestException as exc:
            st.error(f"API unavailable: {exc}")
    if st.button("New chat", use_container_width=True):
        st.session_state.session_id = str(uuid4())
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()


def _send_message(prompt: str, api_url: str, escalation_form: dict | None = None) -> None:
    """Appends the user's turn, calls /chat (optionally carrying a form
    submission), and appends the assistant's reply -- shared by the normal
    chat box and the escalation form's submit button so there's exactly one
    code path that talks to the API."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    payload: dict = {"session_id": st.session_state.session_id, "message": prompt}
    if escalation_form is not None:
        payload["escalation_form"] = escalation_form

    with st.spinner("Checking the HR knowledge base..."):
        try:
            response = requests.post(f"{api_url.rstrip('/')}/chat", json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            data = {
                "reply": f"I could not reach the API: {exc}",
                "citations": [],
                "sources": [],
                "web_citations": [],
                "actions": [],
            }

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["reply"],
            "citations": data.get("citations", []),
            "sources": data.get("sources", []),
            "web_citations": data.get("web_citations", []),
            "actions": data.get("actions", []),
        }
    )


def _render_escalation_form(api_url: str) -> None:
    """The rendered intake form itself (PLAN.md Sec 6.1, Step B). Submitting
    sends a structured escalation_form payload rather than free chat text --
    see src/agent/orchestrator.py::_file_from_form_submission."""
    with st.form(key="escalation_intake_form", clear_on_submit=True):
        st.markdown("**Complaint intake form**")
        category = st.selectbox("Category", _CATEGORY_OPTIONS)
        severity = st.selectbox("Severity", _SEVERITY_OPTIONS)
        description = st.text_area("What happened? (at least 10 characters)")
        parties_raw = st.text_input("Anyone else involved? (comma-separated, optional)")
        incident_date = st.text_input("When did this happen? (optional)")
        desired_outcome = st.text_input("What outcome are you hoping for? (optional)")
        submitted = st.form_submit_button("Submit complaint")

    if not submitted:
        return
    if len(description.strip()) < 10:
        st.error("Please add a bit more detail (at least 10 characters) before submitting.")
        return

    escalation_form = {
        "category": category,
        "severity": severity,
        "description": description.strip(),
        "parties_involved": [p.strip() for p in parties_raw.split(",") if p.strip()],
        "incident_date": incident_date.strip() or None,
        "desired_outcome": desired_outcome.strip() or None,
    }
    _send_message("[submitted the complaint intake form]", api_url, escalation_form=escalation_form)
    st.rerun()


def _render_assistant_details(message: dict, api_url: str, is_latest: bool = False) -> None:
    actions = message.get("actions") or []
    for action in actions:
        if action.get("type") == "escalation_form_required":
            # Only the most recent message ever gets the *interactive* form --
            # once a newer message exists, this request is no longer active.
            if is_latest:
                _render_escalation_form(api_url)
            else:
                st.info(action["label"])
            continue
        if action.get("ticket_id"):
            st.success(f"{action['label']} Ticket: {action['ticket_id']}")
        elif action.get("status") == "pending":
            st.info(action["label"])
        else:
            st.write(action["label"])

    citations = message.get("citations") or []
    if citations:
        with st.expander("Citations", expanded=False):
            for citation in citations:
                st.markdown(
                    f"- `{citation['chunk_id']}` - {citation['title']} / {citation['section_path']}"
                )

    sources = message.get("sources") or []
    if sources:
        with st.expander("Retrieved policy excerpts", expanded=False):
            for source in sources:
                st.markdown(
                    f"**{source['title']} - {source['section_path']}**  \n"
                    f"`{source['chunk_id']}` | similarity `{source['similarity']}`"
                )
                st.caption(source["preview"])

    web_citations = message.get("web_citations") or []
    if web_citations:
        with st.expander("Web sources (DOLE/official)", expanded=False):
            for citation in web_citations:
                st.markdown(f"- [{citation['title']}]({citation['url']})")


for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            is_latest = index == len(st.session_state.messages) - 1
            _render_assistant_details(message, api_url, is_latest=is_latest)

prompt = st.chat_input("Ask an HR policy question")
if prompt:
    _send_message(prompt, api_url)
    st.rerun()
