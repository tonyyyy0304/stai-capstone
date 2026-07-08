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


def _render_assistant_details(message: dict) -> None:
    actions = message.get("actions") or []
    for action in actions:
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


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_assistant_details(message)

prompt = st.chat_input("Ask an HR policy question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {"session_id": st.session_state.session_id, "message": prompt}
    with st.chat_message("assistant"):
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
        st.markdown(data["reply"])
        assistant_message = {
            "role": "assistant",
            "content": data["reply"],
            "citations": data.get("citations", []),
            "sources": data.get("sources", []),
            "web_citations": data.get("web_citations", []),
            "actions": data.get("actions", []),
        }
        _render_assistant_details(assistant_message)
        st.session_state.messages.append(assistant_message)
