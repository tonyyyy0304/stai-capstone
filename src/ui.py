"""Streamlit chat UI for the HR assistant — E.Z.R.A.

Recreates the design in docs/ui-design-revision-request/ (collapsible sidebar,
header "New chat", inline citation/source/web pills, privacy consent gate)
using Streamlit-native widgets + injected CSS, since Streamlit doesn't support
arbitrary client-side component state. Two deliberate deviations from the
static HTML prototype, both driven by Streamlit's rerun model:
  - The privacy gate/declined screen fully replaces the app instead of
    blurring it behind an overlay — this also actually blocks interaction
    with the chat input, not just visually.
  - The sidebar "Recent" list shows only the current session, since there is
    no backend endpoint to list past sessions (fabricating history would be
    misleading).
"""

from __future__ import annotations

import html
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

ACCENT = "#3B6FE0"

GLOBAL_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* Pin light mode regardless of OS/browser dark-mode preference — Streamlit's
   own internal widget chrome (e.g. the chat input's inner wrapper) otherwise
   switches to dark colors that this design doesn't account for. */
:root { color-scheme: light; }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: oklch(85% 0.006 250); border-radius: 8px; }
a { color: __ACCENT__; text-decoration: none; }
a:hover { text-decoration: underline; }

[data-testid="stHeader"] { display: none; }
[data-testid="stAppViewContainer"] { background: oklch(98.2% 0.004 250); }
[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }

/* Sidebar shell */
[data-testid="stSidebar"] {
  background: oklch(96.8% 0.005 250);
  border-right: 1px solid oklch(90% 0.006 250);
  transition: width 0.22s ease, min-width 0.22s ease;
  overflow: hidden;
}
[data-testid="stSidebarContent"] { min-width: 260px; padding: 18px 14px !important; }
[data-testid="stSidebarCollapseButton"] { display: none; }

/* Header hamburger toggle */
.st-key-sidebar_toggle_btn button {
  width: 32px !important; height: 32px !important; min-height: 32px !important;
  border-radius: 7px !important; border: none !important; background: transparent !important;
  color: oklch(40% 0.015 250) !important; font-size: 16px !important; padding: 0 !important;
}
.st-key-sidebar_toggle_btn button:hover { background: oklch(93% 0.008 250) !important; }

/* Header row */
.st-key-header_row {
  padding: 14px 22px; border-bottom: 1px solid oklch(91% 0.006 250);
  background: oklch(98.2% 0.004 250);
}
.st-key-new_chat_wrap { align-items: flex-end !important; }
.st-key-new_chat_btn button {
  padding: 7px 14px !important; border-radius: 8px !important;
  border: 1px solid oklch(88% 0.008 250) !important; background: oklch(99% 0.002 250) !important;
  font-size: 13px !important; font-weight: 500 !important; color: oklch(28% 0.015 255) !important;
  white-space: nowrap !important;
}
.st-key-new_chat_btn button:hover { background: oklch(94% 0.008 250) !important; }

/* Message list */
.st-key-message_list { max-width: 680px; margin: 0 auto; padding: 32px 24px 20px 24px; }
div[class*="st-key-msg_"] { margin-bottom: 30px; }

/* Citation / source / web pills */
div[class*="st-key-pill_"] button {
  font-size: 12px !important; font-weight: 500 !important;
  border-radius: 999px !important; padding: 4px 10px !important;
  white-space: nowrap !important; line-height: 1.4 !important;
}
div[class*="st-key-pill_c_"] button {
  color: __ACCENT__ !important;
  background: color-mix(in oklab, __ACCENT__ 10%, white) !important;
  border: 1px solid color-mix(in oklab, __ACCENT__ 25%, white) !important;
}
div[class*="st-key-pill_s_"] button, div[class*="st-key-pill_w_"] button {
  color: oklch(40% 0.012 250) !important;
  background: oklch(95% 0.006 250) !important;
  border: 1px solid oklch(88% 0.008 250) !important;
}

/* Composer */
[data-testid="stBottomBlockContainer"] {
  background: oklch(98.2% 0.004 250) !important; padding-bottom: 28px !important;
}
[data-testid="stChatInput"] {
  max-width: 680px; margin: 0 auto;
  border: 1px solid oklch(87% 0.008 250) !important; border-radius: 16px !important;
  background: oklch(99.3% 0.002 250) !important;
  box-shadow: 0 1px 2px oklch(0% 0 0 / 0.04);
}
/* Streamlit's inner wrapper + textarea pick up its dark-theme colors when
   the OS/browser prefers dark mode (also draws a theme-red busy-state
   border while a submission is processing); the outer stChatInput already
   supplies this design's border/background, so force both to match it. */
[data-testid="stChatInput"] > div {
  border: none !important; background: oklch(99.3% 0.002 250) !important;
}
[data-testid="stChatInputTextArea"] {
  font-size: 14.5px !important; background: transparent !important;
  color: oklch(20% 0.015 255) !important; caret-color: oklch(20% 0.015 255) !important;
}
[data-testid="stChatInputTextArea"]::placeholder { color: oklch(50% 0.012 250); }
[data-testid="stChatInputSubmitButton"] {
  background: __ACCENT__ !important; border-radius: 9px !important; color: white !important;
}
[data-testid="stChatInputInstructions"] { display: none; }
[data-testid="stBottomBlockContainer"]::after {
  content: "Responses are grounded in HR policy documents and may be reviewed by HR staff.";
  display: block; max-width: 680px; margin: 6px auto 0 auto;
  font-size: 11.5px; color: oklch(60% 0.01 250); text-align: center;
}

/* Privacy gate */
.st-key-gate_card {
  width: 100%; max-width: 440px; margin: 0 auto;
  background: oklch(99.3% 0.002 250); border-radius: 16px;
  border: 1px solid oklch(90% 0.006 250); box-shadow: 0 20px 50px oklch(0% 0 0 / 0.18);
  padding: 28px 28px 24px 28px;
  display: flex !important; flex-direction: column !important; gap: 16px !important;
}
.st-key-privacy_agree_btn button {
  background: __ACCENT__ !important; color: white !important; border: none !important;
  border-radius: 9px !important; padding: 10px 16px !important;
  font-size: 14px !important; font-weight: 500 !important;
}
.st-key-privacy_decline_btn button {
  background: transparent !important; color: oklch(38% 0.014 250) !important;
  border: 1px solid oklch(88% 0.008 250) !important; border-radius: 9px !important;
  padding: 10px 16px !important; font-size: 14px !important; font-weight: 500 !important;
}

/* Declined screen */
.st-key-declined_card {
  width: 100%; max-width: 420px; margin: 0 auto;
  display: flex !important; flex-direction: column !important;
  align-items: center !important; text-align: center; gap: 14px !important;
}
.st-key-privacy_review_btn button {
  padding: 9px 18px !important; border-radius: 9px !important;
  border: 1px solid oklch(88% 0.008 250) !important; background: oklch(99% 0.002 250) !important;
  color: oklch(28% 0.015 255) !important; font-size: 14px !important; font-weight: 500 !important;
}

/* Escalation intake form (rendered inline under a pending action label) */
[data-testid="stForm"] {
  border: 1px solid oklch(90% 0.006 250) !important;
  border-radius: 12px !important;
  background: oklch(97% 0.004 250) !important;
  padding: 16px !important;
}
[data-testid="stFormSubmitButton"] button {
  background: __ACCENT__ !important; color: white !important; border: none !important;
  border-radius: 8px !important; font-weight: 500 !important;
}

/* Typing indicator -- rendered in-flow in the message list so a pending
   reply looks like part of the conversation instead of a generic spinner
   stuck at the page's left edge. */
.st-key-typing_indicator { max-width: 680px; margin: 0 auto; padding: 0 24px 20px 24px; }
@keyframes ezra-typing-bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
  30% { transform: translateY(-3px); opacity: 1; }
}
.ezra-typing-dot {
  width: 6px; height: 6px; border-radius: 999px; background: oklch(55% 0.012 250);
  display: inline-block; animation: ezra-typing-bounce 1.1s ease-in-out infinite;
}
.ezra-typing-dot:nth-child(2) { animation-delay: 0.12s; }
.ezra-typing-dot:nth-child(3) { animation-delay: 0.24s; }

/* Quick-start prompt chips shown before the user's first message */
.st-key-quick_prompts { max-width: 680px; margin: 0 auto; padding: 0 24px 20px 62px; }
div[class*="st-key-chip_"] button {
  font-size: 13px !important; font-weight: 500 !important; text-align: left !important;
  color: oklch(30% 0.015 255) !important; background: oklch(99% 0.002 250) !important;
  border: 1px solid oklch(88% 0.008 250) !important; border-radius: 10px !important;
  padding: 9px 12px !important; white-space: normal !important; line-height: 1.35 !important;
  width: 100% !important;
}
div[class*="st-key-chip_"] button:hover {
  background: color-mix(in oklab, __ACCENT__ 8%, white) !important;
  border-color: color-mix(in oklab, __ACCENT__ 30%, white) !important;
}
</style>
""".replace("__ACCENT__", ACCENT)

QUICK_PROMPTS = [
    "Is 13th month pay required?",
    "Who can I file a complaint to?",
    "How many vacation leave credits do I get?",
    "What's the process for reporting harassment?",
]


def _greeting_message() -> dict:
    return {
        "role": "assistant",
        "content": "Hi. Ask me about HR policies, or tell me if you need to start a complaint.",
        "citations": [],
        "sources": [],
        "web_citations": [],
        "actions": [],
        "token_usage": {},
    }


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = [_greeting_message()]
    if "privacy_status" not in st.session_state:
        st.session_state.privacy_status = "pending"  # pending | agreed | declined
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True
    if "expanded_panels" not in st.session_state:
        st.session_state.expanded_panels = {}
    st.session_state.setdefault("api_url", config.API_URL)
    st.session_state.setdefault("awaiting_response", False)
    st.session_state.setdefault("pending_request", None)


def _toggle_sidebar() -> None:
    st.session_state.sidebar_open = not st.session_state.sidebar_open


def _start_new_chat() -> None:
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = [_greeting_message()]
    st.session_state.expanded_panels = {}
    st.session_state.awaiting_response = False
    st.session_state.pending_request = None


def _accept_privacy() -> None:
    st.session_state.privacy_status = "agreed"


def _decline_privacy() -> None:
    st.session_state.privacy_status = "declined"


def _reconsider_privacy() -> None:
    st.session_state.privacy_status = "pending"


def _toggle_panel(i: int, panel_type: str) -> None:
    panels = st.session_state.expanded_panels.setdefault(
        i, {"citations": False, "sources": False, "web": False}
    )
    panels[panel_type] = not panels[panel_type]


def _sidebar_width_css() -> str:
    width = 260 if st.session_state.sidebar_open else 0
    # When collapsed, the inner content stays at its fixed 260px width (so it
    # doesn't reflow) and only the outer width goes to 0 with overflow
    # clipped — but the clipped-away content still sits in the layout and
    # intercepts clicks on the header behind it unless pointer-events is
    # explicitly turned off too.
    closed_extra = "" if st.session_state.sidebar_open else "border-right: none !important; pointer-events: none !important;"
    return (
        "<style>[data-testid=\"stSidebar\"] { "
        f"width: {width}px !important; min-width: {width}px !important; {closed_extra} "
        "}</style>"
    )


def _gate_background_css(privacy_status: str) -> str:
    bg = "oklch(20% 0.01 255 / 0.42)" if privacy_status == "pending" else "oklch(98.2% 0.004 250)"
    return (
        "<style>"
        f'[data-testid="stAppViewContainer"] {{ background: {bg} !important; }}'
        '[data-testid="stMainBlockContainer"] {'
        "  min-height: 100vh !important; display: flex !important;"
        "  align-items: center !important; justify-content: center !important;"
        "  padding: 24px !important; max-width: 100% !important;"
        "}"
        "</style>"
    )


def _render_privacy_gate(accent: str) -> None:
    with st.container(key="gate_card"):
        st.markdown(
            f'<div style="width:32px;height:32px;border-radius:8px;background:{accent};"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:17px;font-weight:600;color:oklch(20% 0.015 255);">'
            "Before you continue</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:14px;line-height:1.6;color:oklch(38% 0.014 250);">'
            "E.Z.R.A. can help with HR policy questions and complaint intake. If you file a "
            "complaint, details you share — including personal information about you or "
            "others — are stored so HR can follow up and, where required, escalated to a "
            "human reviewer.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:14px;line-height:1.6;color:oklch(38% 0.014 250);'
            'font-weight:500;">Do you consent to that information being stored for '
            "complaint handling?</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                "I agree, continue",
                key="privacy_agree_btn",
                on_click=_accept_privacy,
                use_container_width=True,
            )
        with col2:
            st.button(
                "Decline",
                key="privacy_decline_btn",
                on_click=_decline_privacy,
                use_container_width=True,
            )


def _render_declined_screen() -> None:
    with st.container(key="declined_card"):
        st.markdown(
            '<div style="font-size:17px;font-weight:600;color:oklch(20% 0.015 255);">'
            "Consent needed to continue</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:14px;line-height:1.6;color:oklch(45% 0.012 250);">'
            "E.Z.R.A. can't take complaint details or store your messages without your "
            "consent. You can review the notice again if you'd like to proceed.</div>",
            unsafe_allow_html=True,
        )
        st.button("Review the notice again", key="privacy_review_btn", on_click=_reconsider_privacy)


def _render_sidebar(accent: str, dev_mode: bool) -> None:
    with st.sidebar:
        st.markdown(
            f'''<div style="display:flex;flex-direction:column;min-width:260px;">
  <div style="display:flex;align-items:center;gap:8px;padding:6px 8px 18px 8px;">
    <div style="width:22px;height:22px;border-radius:6px;background:{accent};flex-shrink:0;"></div>
    <div style="font-size:14px;font-weight:600;letter-spacing:0.01em;color:oklch(20% 0.015 255);">E.Z.R.A.</div>
  </div>
  <div style="font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.06em;color:oklch(55% 0.012 250);padding:4px 8px 8px 8px;">Recent</div>
  <div style="display:flex;flex-direction:column;gap:1px;padding:9px 8px;border-radius:8px;background:oklch(92% 0.012 250);">
    <div style="font-size:13.5px;font-weight:500;color:oklch(22% 0.015 255);">Current chat</div>
    <div style="font-size:11.5px;color:oklch(56% 0.012 250);">Today</div>
  </div>
</div>''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''<div style="display:flex;align-items:center;gap:8px;padding:10px 8px;border-top:1px solid oklch(90% 0.006 250);margin-top:10px;">
  <div style="width:26px;height:26px;border-radius:999px;background:oklch(88% 0.01 250);flex-shrink:0;"></div>
  <div style="font-size:12.5px;color:oklch(45% 0.012 250);">Employee session</div>
</div>''',
            unsafe_allow_html=True,
        )

        if dev_mode:
            with st.expander("Developer tools", expanded=False):
                st.session_state.api_url = st.text_input(
                    "API URL", value=st.session_state.api_url
                )
                if st.button("Check API", key="check_api_btn"):
                    try:
                        health = requests.get(
                            f"{st.session_state.api_url.rstrip('/')}/health", timeout=10
                        )
                        if health.ok:
                            st.success(f"API status: {health.json()['status']}")
                        else:
                            st.error(f"API returned {health.status_code}")
                    except requests.RequestException as exc:
                        st.error(f"API unavailable: {exc}")
                if st.button("Refresh usage", key="refresh_usage_btn"):
                    try:
                        usage_resp = requests.get(
                            f"{st.session_state.api_url.rstrip('/')}/usage", timeout=10
                        )
                        usage_resp.raise_for_status()
                        usage_data = usage_resp.json()
                        today, all_time = usage_data["today"], usage_data["all_time"]
                        col1, col2 = st.columns(2)
                        col1.metric("Requests today", today.get("request_count", 0))
                        col2.metric("Tokens today", today.get("total_tokens", 0))
                        col1.metric("Requests all-time", all_time.get("request_count", 0))
                        col2.metric("Tokens all-time", all_time.get("total_tokens", 0))
                    except requests.RequestException as exc:
                        st.error(f"Could not fetch usage: {exc}")


def _render_header() -> None:
    with st.container(key="header_row"):
        col1, col2, col3 = st.columns([0.06, 0.6, 0.34], vertical_alignment="center")
        with col1:
            st.button("☰", key="sidebar_toggle_btn", on_click=_toggle_sidebar)
        with col2:
            st.markdown(
                '<div style="display:flex;flex-direction:column;gap:1px;">'
                '<div style="font-size:15px;font-weight:600;color:oklch(24% 0.015 255);">E.Z.R.A.</div>'
                '<div style="font-size:12px;color:oklch(52% 0.012 250);">Grounded HR policy answers, with citations</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        with col3:
            with st.container(key="new_chat_wrap"):
                st.button("+  New chat", key="new_chat_btn", on_click=_start_new_chat)


def _render_actions(actions: list[dict], is_latest: bool = False) -> None:
    for action in actions:
        label = html.escape(action.get("label", ""))
        status = action.get("status", "completed")
        ticket_id = action.get("ticket_id")
        if action.get("type") == "escalation_form_required":
            # Only the most recent message ever gets the *interactive* form --
            # once a newer message exists, this request is no longer active.
            if is_latest:
                _render_escalation_form()
            else:
                st.markdown(
                    f'<div style="font-size:13px;color:oklch(45% 0.012 250);margin-top:8px;">{label}</div>',
                    unsafe_allow_html=True,
                )
            continue
        if status == "completed" and ticket_id:
            ticket_html = (
                '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                f'color:oklch(38% 0.06 155);">{html.escape(ticket_id)}</span>'
            )
            st.markdown(
                '<div style="display:flex;align-items:center;gap:8px;padding:9px 12px;'
                "border-radius:9px;background:oklch(94% 0.03 155);"
                'border:1px solid oklch(85% 0.05 155);margin-top:8px;">'
                '<div style="width:16px;height:16px;border-radius:999px;background:oklch(52% 0.13 155);'
                'flex-shrink:0;display:flex;align-items:center;justify-content:center;">'
                '<div style="width:6px;height:6px;border-radius:999px;background:oklch(99% 0 0);"></div>'
                "</div>"
                f'<div style="font-size:13px;color:oklch(28% 0.04 155);">{label} {ticket_html}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        elif status == "pending":
            st.markdown(
                '<div style="display:flex;align-items:center;gap:8px;padding:9px 12px;'
                "border-radius:9px;background:oklch(96% 0.03 85);"
                'border:1px solid oklch(87% 0.05 85);margin-top:8px;">'
                '<div style="width:16px;height:16px;border-radius:999px;background:oklch(65% 0.13 85);'
                'flex-shrink:0;display:flex;align-items:center;justify-content:center;">'
                '<div style="width:6px;height:6px;border-radius:999px;background:oklch(99% 0 0);"></div>'
                "</div>"
                f'<div style="font-size:13px;color:oklch(35% 0.05 85);">{label}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="font-size:13px;color:oklch(45% 0.012 250);margin-top:8px;">{label}</div>',
                unsafe_allow_html=True,
            )


def _queue_message(prompt: str, escalation_form: dict | None = None) -> None:
    """Appends the user's turn and queues the API call for the next script
    run. Shared by the composer, the quick-start chips, and the escalation
    form's submit button so there's exactly one code path that talks to the
    API. Split from the actual request (see `_fetch_pending_response`) so a
    typing indicator can render in-flow *before* the blocking network call,
    instead of a spinner appearing outside the message list."""
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "citations": [],
            "sources": [],
            "web_citations": [],
            "actions": [],
            "token_usage": {},
        }
    )
    st.session_state.pending_request = {"message": prompt, "escalation_form": escalation_form}
    st.session_state.awaiting_response = True


def _fetch_pending_response() -> None:
    """Performs the queued API call and appends the assistant's reply. Must
    only be called after the typing indicator has already been rendered."""
    pending = st.session_state.pending_request or {}
    payload: dict = {
        "session_id": st.session_state.session_id,
        "message": pending.get("message", ""),
    }
    if pending.get("escalation_form") is not None:
        payload["escalation_form"] = pending["escalation_form"]

    try:
        response = requests.post(
            f"{st.session_state.api_url.rstrip('/')}/chat", json=payload, timeout=90
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        data = {
            "reply": f"I could not reach the API: {exc}",
            "citations": [],
            "sources": [],
            "web_citations": [],
            "actions": [],
            "token_usage": {},
        }

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["reply"],
            "citations": data.get("citations", []),
            "sources": data.get("sources", []),
            "web_citations": data.get("web_citations", []),
            "actions": data.get("actions", []),
            "token_usage": data.get("token_usage", {}),
        }
    )
    st.session_state.pending_request = None
    st.session_state.awaiting_response = False


def _render_typing_indicator(accent: str) -> None:
    with st.container(key="typing_indicator"):
        st.markdown(
            '<div style="display:flex;gap:10px;align-items:center;">'
            f'<div style="width:26px;height:26px;border-radius:7px;background:{accent};'
            'flex-shrink:0;display:flex;align-items:center;justify-content:center;">'
            '<div style="width:8px;height:8px;border-radius:2px;background:oklch(99% 0 0);"></div>'
            "</div>"
            '<div style="display:flex;gap:4px;align-items:center;padding:9px 2px;">'
            '<span class="ezra-typing-dot"></span>'
            '<span class="ezra-typing-dot"></span>'
            '<span class="ezra-typing-dot"></span>'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_quick_prompts() -> None:
    """Suggested starter prompts shown only before the conversation has
    actually started, so returning users mid-conversation aren't shown
    stale suggestions above the composer."""
    if len(st.session_state.messages) != 1 or st.session_state.awaiting_response:
        return
    with st.container(key="quick_prompts"):
        cols = st.columns(2)
        for idx, prompt in enumerate(QUICK_PROMPTS):
            with cols[idx % 2]:
                if st.button(prompt, key=f"chip_{idx}", use_container_width=True):
                    _queue_message(prompt)
                    st.rerun()


def _render_escalation_form() -> None:
    """The rendered intake form itself (PLAN.md Sec 6.1, Step B). Submitting
    sends a structured escalation_form payload rather than free chat text --
    see src/agent/orchestrator.py::_file_from_form_submission."""
    with st.container(key="escalation_form_wrap"):
        with st.form(key="escalation_intake_form", clear_on_submit=True):
            st.markdown(
                '<div style="font-size:13.5px;font-weight:600;color:oklch(24% 0.015 255);'
                'margin-bottom:6px;">Complaint intake form</div>',
                unsafe_allow_html=True,
            )
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
    _queue_message("[submitted the complaint intake form]", escalation_form=escalation_form)
    st.rerun()


def _render_pills_and_panels(i: int, msg: dict, accent: str) -> None:
    citations = msg.get("citations") or []
    sources = msg.get("sources") or []
    web_citations = msg.get("web_citations") or []
    if not (citations or sources or web_citations):
        return

    panels = st.session_state.expanded_panels.setdefault(
        i, {"citations": False, "sources": False, "web": False}
    )

    pill_specs = []
    if citations:
        pill_specs.append(("citations", f"Citations {len(citations)}", f"pill_c_{i}"))
    if sources:
        pill_specs.append(("sources", f"Sources {len(sources)}", f"pill_s_{i}"))
    if web_citations:
        pill_specs.append(("web", f"Web sources {len(web_citations)}", f"pill_w_{i}"))

    ratios = [0.16] * len(pill_specs) + [max(0.1, 1 - 0.16 * len(pill_specs))]
    cols = st.columns(ratios)
    for col, (panel_type, label, key) in zip(cols, pill_specs):
        with col:
            st.button(label, key=key, on_click=_toggle_panel, args=(i, panel_type))

    if panels["citations"] and citations:
        rows = []
        for n, c in enumerate(citations, start=1):
            title = html.escape(c["title"])
            section = html.escape(c["section_path"])
            chunk_id = html.escape(c["chunk_id"])
            rows.append(
                '<div style="font-size:12.5px;color:oklch(35% 0.014 250);display:flex;gap:6px;">'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;color:{accent};flex-shrink:0;">[{n}]</span>'
                f'<span><span style="font-weight:500;">{title}</span> — {section} '
                '<span style="font-family:\'IBM Plex Mono\',monospace;color:oklch(55% 0.012 250);'
                f'font-size:11.5px;">{chunk_id}</span></span>'
                "</div>"
            )
        st.markdown(
            '<div style="display:flex;flex-direction:column;gap:6px;padding:10px 12px;'
            "border-radius:10px;background:oklch(97% 0.004 250);"
            'border:1px solid oklch(90% 0.006 250);margin-top:8px;">' + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )

    if panels["sources"] and sources:
        rows = []
        for s in sources:
            title = html.escape(s["title"])
            section = html.escape(s["section_path"])
            preview = html.escape(s["preview"])
            similarity = s.get("similarity", 0)
            rows.append(
                '<div style="display:flex;flex-direction:column;gap:4px;">'
                '<div style="font-size:12.5px;font-weight:500;color:oklch(28% 0.015 255);'
                'display:flex;justify-content:space-between;gap:10px;">'
                f"<span>{title} · {section}</span>"
                '<span style="font-family:\'IBM Plex Mono\',monospace;font-weight:400;font-size:11px;'
                f'color:oklch(56% 0.012 250);white-space:nowrap;flex-shrink:0;">sim {similarity:.2f}</span>'
                "</div>"
                f'<div style="font-size:12.5px;color:oklch(48% 0.012 250);line-height:1.5;margin-top:1px;">{preview}</div>'
                "</div>"
            )
        st.markdown(
            '<div style="display:flex;flex-direction:column;gap:10px;padding:12px 14px;'
            "border-radius:10px;background:oklch(97% 0.004 250);"
            'border:1px solid oklch(90% 0.006 250);margin-top:8px;">' + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )

    if panels["web"] and web_citations:
        rows = []
        for w in web_citations:
            title = html.escape(w["title"])
            url = w.get("url", "")
            if url.startswith("http://") or url.startswith("https://"):
                safe_url = html.escape(url, quote=True)
                rows.append(
                    f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
                    f'style="font-size:12.5px;color:{accent};text-decoration:none;">{title} ↗</a>'
                )
            else:
                rows.append(f'<div style="font-size:12.5px;color:oklch(40% 0.012 250);">{title}</div>')
        st.markdown(
            '<div style="display:flex;flex-direction:column;gap:6px;padding:10px 12px;'
            "border-radius:10px;background:oklch(97% 0.004 250);"
            'border:1px solid oklch(90% 0.006 250);margin-top:8px;">' + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )


def _render_message(i: int, msg: dict, accent: str, is_latest: bool = False) -> None:
    with st.container(key=f"msg_{i}"):
        content = html.escape(msg["content"])
        if msg["role"] == "assistant":
            st.markdown(
                '<div style="display:flex;gap:10px;align-items:flex-start;">'
                f'<div style="width:26px;height:26px;border-radius:7px;background:{accent};'
                'flex-shrink:0;margin-top:2px;display:flex;align-items:center;justify-content:center;">'
                '<div style="width:8px;height:8px;border-radius:2px;background:oklch(99% 0 0);"></div>'
                "</div>"
                '<div style="font-size:15px;line-height:1.55;color:oklch(20% 0.015 255);'
                f'white-space:pre-wrap;min-width:0;">{content}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            _render_actions(msg.get("actions") or [], is_latest=is_latest)
            _render_pills_and_panels(i, msg, accent)
        else:
            st.markdown(
                '<div style="display:flex;justify-content:flex-end;">'
                '<div style="font-size:15px;line-height:1.55;padding:11px 15px;border-radius:14px;'
                f'background:oklch(94% 0.02 250);color:oklch(20% 0.015 255);'
                f'white-space:pre-wrap;max-width:78%;">{content}</div>'
                "</div>",
                unsafe_allow_html=True,
            )


def _render_messages(accent: str) -> None:
    with st.container(key="message_list"):
        n = len(st.session_state.messages)
        for i, message in enumerate(st.session_state.messages):
            _render_message(i, message, accent, is_latest=(i == n - 1))


st.set_page_config(page_title="E.Z.R.A.", page_icon="💬", layout="wide")
_init_state()

_dev_mode = st.query_params.get("dev") == "1"
_accent = ACCENT

st.html(GLOBAL_CSS)
st.html(_sidebar_width_css())

_privacy_status = st.session_state.privacy_status
if _privacy_status != "agreed":
    st.html(_gate_background_css(_privacy_status))
    if _privacy_status == "pending":
        _render_privacy_gate(_accent)
    else:
        _render_declined_screen()
    st.stop()

_render_sidebar(_accent, _dev_mode)
_render_header()
_render_messages(_accent)

if st.session_state.awaiting_response:
    # Render the typing indicator first so it's flushed to the browser
    # in-flow (matching the assistant-message layout) before the blocking
    # network call below, rather than a spinner floating outside the
    # message list.
    _render_typing_indicator(_accent)
    _fetch_pending_response()
    st.rerun()

_render_quick_prompts()

_prompt = st.chat_input("Ask an HR policy question…")
if _prompt:
    _queue_message(_prompt)
    st.rerun()