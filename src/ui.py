"""Streamlit chat UI for the assistant (branding from config.ASSISTANT_NAME).

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
from datetime import date
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config

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
/* stMain is a flex sibling of stSidebar under stAppViewContainer. Flex
   children default to min-width:auto, so stMain refuses to shrink below its
   content's intrinsic width even with flex:1 -- if anything inside (e.g. the
   centered 680px message column) has a wide-enough intrinsic min-content
   width, the whole row overflows the viewport instead of stMain actually
   shrinking to fit beside the sidebar. Without this, the header (which has
   no such forcing content) sits correctly while the message list/quick
   prompts/composer -- which all center via max-width+margin:auto -- end up
   centered within an oversized box and pushed off-screen right, with a
   horizontal scrollbar as the visible symptom. */
[data-testid="stMain"] { min-width: 0 !important; overflow-x: hidden !important; }

/* Sidebar shell */
[data-testid="stSidebar"] {
  background: oklch(96.8% 0.005 250);
  border-right: 1px solid oklch(90% 0.006 250);
  transition: width 0.22s ease, min-width 0.22s ease;
  overflow: hidden;
}
/* box-sizing:border-box is the fix here, not the min-width/padding values
   themselves -- by default (content-box) a 260px min-width PLUS 14px
   padding on each side renders at 288px actual width, 28px wider than the
   parent stSidebar's own fixed 260px (_sidebar_width_css()). Since stSidebar
   has overflow:hidden, that 28px of overflow was silently clipped off the
   right edge of every row -- and since only the right side gets cut, the
   visible content reads as unevenly balanced/off-center, not just clipped. */
[data-testid="stSidebarContent"] { min-width: 260px; padding: 18px 14px !important; box-sizing: border-box; }
[data-testid="stSidebarCollapseButton"] { display: none; }
/* Streamlit's own floating re-expand chevron -- auto-appears top-left
   whenever stSidebar's width hits 0, which our collapsed state does.
   Left visible, it overlaps the custom hamburger toggle below and steals
   its clicks, so a click meant for our button re-expands via Streamlit's
   own untracked mechanism instead of flipping session_state.sidebar_open --
   the two state sources desync and the sidebar gets stuck. */
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

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

/* Document upload submit button (Component 14) -- same accent-filled
   pattern as .st-key-privacy_agree_btn below. */
.st-key-upload_submit_btn button {
  background: __ACCENT__ !important; color: white !important; border: none !important;
  border-radius: 9px !important; padding: 9px 16px !important;
  font-size: 13.5px !important; font-weight: 500 !important; margin-top: 4px !important;
}
.st-key-upload_submit_btn button:hover {
  background: color-mix(in oklab, __ACCENT__ 88%, black) !important;
}

/* Document uploader (Component 14) -- lives in the main chat column, not
   the sidebar (which is checklist-only), aligned to the same centered
   width as the message list below it. */
.st-key-uploader_section { max-width: 680px; margin: 0 auto; padding: 16px 24px 0 24px; }

/* Restyles Streamlit's built-in st.spinner (shown during POST /upload-doc)
   to match this design's accent/type instead of Streamlit's stock red-ish
   default -- the icon's stroke follows `color` via currentColor, so setting
   color here recolors both the spinner icon and its text in one rule. */
.st-key-uploader_section [data-testid="stSpinner"] {
  color: __ACCENT__ !important;
  font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.st-key-uploader_section [data-testid="stSpinner"] p { font-size: 13.5px !important; }

/* Message list */
.st-key-message_list { max-width: 680px; margin: 0 auto; padding: 32px 24px 20px 24px; }
div[class*="st-key-msg_"] { margin-bottom: 30px; }

/* Assistant message: avatar + Markdown body laid out as a flex row. The body is
   a real st.markdown() block (so bold/lists/tables render), not escaped text. */
div[class*="st-key-msg_row_"] [data-testid="stVerticalBlock"] {
  flex-direction: row; gap: 10px; align-items: flex-start;
}
div[class*="st-key-msg_row_"] [data-testid="stVerticalBlock"] > div:first-child { flex: 0 0 auto; }
div[class*="st-key-msg_row_"] [data-testid="stVerticalBlock"] > div:last-child { flex: 1 1 auto; min-width: 0; }
div[class*="st-key-msg_row_"] [data-testid="stMarkdownContainer"] {
  font-size: 15px; line-height: 1.55; color: oklch(20% 0.015 255);
}
div[class*="st-key-msg_row_"] [data-testid="stMarkdownContainer"] p:first-child { margin-top: 0; }
div[class*="st-key-msg_row_"] [data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0; }
div[class*="st-key-msg_row_"] [data-testid="stMarkdownContainer"] ul,
div[class*="st-key-msg_row_"] [data-testid="stMarkdownContainer"] ol { margin: 6px 0; padding-left: 22px; }
div[class*="st-key-msg_row_"] [data-testid="stMarkdownContainer"] li { margin: 2px 0; }
div[class*="st-key-msg_row_"] [data-testid="stMarkdownContainer"] table {
  border-collapse: collapse; font-size: 13.5px; margin: 8px 0;
}
div[class*="st-key-msg_row_"] [data-testid="stMarkdownContainer"] th,
div[class*="st-key-msg_row_"] [data-testid="stMarkdownContainer"] td {
  border: 1px solid oklch(90% 0.006 250); padding: 6px 10px; text-align: left;
}
div[class*="st-key-msg_row_"] [data-testid="stMarkdownContainer"] th {
  background: oklch(97% 0.004 250); font-weight: 600;
}

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
  content: "Responses are grounded in the DLSU Faculty Manual and official onboarding documents; always verify with your college's HR office.";
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
    "What are the stages of the DLSU faculty pre-boarding process?",
    "Which documents do I submit after a conditional job offer?",
    "What are the pre-employment requirements for new faculty?",
    "Do I need an NBI clearance to start?",
]


def _greeting_message() -> dict:
    return {
        "role": "assistant",
        "content": "Hi. Ask me about faculty onboarding, pre-employment requirements, or the DLSU Faculty Manual.",
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
    # Component 14: needed so "what's my document status?" in chat can
    # resolve to a checklist lookup, and so the uploader below knows who
    # it's uploading for.
    st.session_state.setdefault("employee_id", "")
    # Hidden until the user opens it -- either the sidebar's "Verify a
    # document" button (_open_document_flow) or an assistant message
    # carrying an "unlock_document_flow" action (Intent.DOCUMENT_UPLOAD/
    # DOCUMENT_STATUS, src/agent/orchestrator.py). Persists for the rest of
    # the session once opened (by design, not an oversight) -- an unrelated
    # question later shouldn't re-hide it.
    st.session_state.setdefault("show_upload_flow", False)
    # Remembered across both documents' uploads so the user doesn't retype
    # identity fields for the second one -- shared by both cards.
    st.session_state.setdefault("upload_full_name", "")
    st.session_state.setdefault("upload_dob", "")
    # Faculty class (audience-class slug, config.AUDIENCE_ORDER) — persisted
    # server-side via set_faculty_class() on first submit, but remembered
    # here too so the second document's upload doesn't need it re-picked.
    st.session_state.setdefault("upload_faculty_class", config.AUDIENCE_ORDER[0])
    # Deferred-render pattern (matches pending_request/awaiting_response
    # above) -- set on submit, rendered on the NEXT run, so the result
    # banner survives the rerun that follows a successful upload. Keyed by
    # doc_type so each document's card owns its own banner instead of one
    # upload's result bleeding onto the other document's card.
    st.session_state.setdefault("last_upload_result", {})


def _toggle_sidebar() -> None:
    st.session_state.sidebar_open = not st.session_state.sidebar_open


def _open_document_flow() -> None:
    st.session_state.show_upload_flow = True


def _start_new_chat() -> None:
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = [_greeting_message()]
    st.session_state.expanded_panels = {}
    st.session_state.awaiting_response = False
    st.session_state.pending_request = None
    st.session_state.show_upload_flow = False  # re-lock; a new conversation hasn't asked for it yet


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
            "This assistant can help with faculty onboarding, pre-employment requirements, "
            "and DLSU Faculty Manual questions. Your messages are stored to maintain "
            "conversation context. If you submit a document (e.g. NBI Clearance, government ID) "
            "for verification, it is processed and its extracted fields are validated.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:14px;line-height:1.6;color:oklch(38% 0.014 250);'
            'font-weight:500;">Do you consent to that information being stored?</div>',
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
            f"{config.ASSISTANT_NAME} can't store your messages without your consent. You can review the "
            "notice again if you'd like to proceed.</div>",
            unsafe_allow_html=True,
        )
        st.button("Review the notice again", key="privacy_review_btn", on_click=_reconsider_privacy)


_DOC_TYPE_LABELS = {"nbi_clearance": "NBI Clearance", "government_id": "Government ID"}

# Status/outcome -> (text color, background, border), extending this file's
# existing oklch language. "needs_review" reuses the exact amber already
# defined for _render_actions' "pending" chip rather than inventing a
# separate token; "accepted"/"validated" and "rejected"/"needs_review" are
# aliased together since ChecklistStatus (DocStatus) and the /upload-doc
# result (ValidationOutcome) use different vocabularies for the same idea.
_STATUS_STYLES = {
    "validated": ("oklch(35% 0.12 145)", "oklch(96% 0.03 145)", "oklch(85% 0.06 145)"),
    "accepted": ("oklch(35% 0.12 145)", "oklch(96% 0.03 145)", "oklch(85% 0.06 145)"),
    "needs_review": ("oklch(45% 0.11 85)", "oklch(96% 0.03 85)", "oklch(87% 0.05 85)"),
    "rejected": ("oklch(45% 0.15 25)", "oklch(96% 0.03 25)", "oklch(87% 0.06 25)"),
    "submitted": ("oklch(40% 0.012 250)", "oklch(95% 0.006 250)", "oklch(88% 0.008 250)"),
    "missing": ("oklch(55% 0.012 250)", "oklch(96% 0.004 250)", "oklch(90% 0.006 250)"),
}


def _is_valid_iso_date(text: str) -> bool:
    """Real calendar validation (rejects 2023-13-40), not just a regex shape
    check -- catches a malformed DOB before it's spent on a POST /upload-doc
    round trip, matching the same discipline as the rest of this project
    (fail before the network call, not after)."""
    try:
        date.fromisoformat(text)
        return True
    except ValueError:
        return False


def _status_badge_style(status: str) -> tuple[str, str, str]:
    return _STATUS_STYLES.get(status, _STATUS_STYLES["missing"])


def _status_badge_html(status: str) -> str:
    text_color, bg_color, border_color = _status_badge_style(status)
    label = status.replace("_", " ").title()
    return (
        f'<span style="font-size:11px;font-weight:500;padding:3px 9px;border-radius:999px;'
        f'color:{text_color};background:{bg_color};border:1px solid {border_color};'
        f'white-space:nowrap;">{label}</span>'
    )


def _fetch_checklist() -> dict | None:
    """GET /onboarding-status (+ conditional GET /hr-notifications) -- called
    once per script run from the top-level flow, after the sidebar's
    Employee ID input has run, and shared by both the sidebar checklist and
    the main-column document cards so neither fetches it twice. A cheap
    SQLite read, no LLM/vision cost, so it reflects the latest state on
    every rerun (including the one that follows a successful upload)
    without needing a manual page refresh.

    Returns None when there's no employee ID yet (nothing to show), or
    {"error": True} on a request failure (still something to show: an error
    line) -- kept distinct from None so callers don't conflate "not started"
    with "failed"."""
    employee_id = st.session_state.employee_id.strip()
    if not employee_id:
        return None

    try:
        response = requests.get(
            f"{st.session_state.api_url.rstrip('/')}/onboarding-status/{employee_id}", timeout=10
        )
        response.raise_for_status()
        checklist = response.json()
    except requests.RequestException:
        return {"error": True}

    documents = checklist.get("documents", [])
    validated_count = sum(1 for d in documents if d["status"] == "validated")
    sent_to_hr = False
    if documents and validated_count == len(documents):
        # Only worth asking once the checklist is actually complete -- this
        # is the same gate src/api.py uses to decide whether it ever queued
        # a send (checklist.missing == []).
        try:
            hr_response = requests.get(
                f"{st.session_state.api_url.rstrip('/')}/hr-notifications/{employee_id}", timeout=10
            )
            hr_response.raise_for_status()
            sent_to_hr = hr_response.json().get("sent", False)
        except requests.RequestException:
            pass
    checklist["sent_to_hr"] = sent_to_hr
    return checklist


def _doc_status(checklist: dict | None, doc_type: str) -> str:
    """Status badge input for a single document card. Missing checklist
    (no employee ID yet, or a failed fetch) and a doc_type with no matching
    entry both read the same as "missing" -- there's nothing more specific
    to say in either case."""
    if not checklist or checklist.get("error"):
        return "missing"
    for doc in checklist.get("documents", []):
        if doc["doc_type"] == doc_type:
            return doc["status"]
    return "missing"


def _render_sidebar_checklist(checklist: dict | None) -> None:
    """Renders the sidebar's compact checklist card from an already-fetched
    checklist (see _fetch_checklist). No I/O here."""
    if checklist is None:
        return
    if checklist.get("error"):
        st.markdown(
            '<div style="font-size:12px;color:oklch(55% 0.012 250);padding:4px 0;">'
            "Could not load document checklist.</div>",
            unsafe_allow_html=True,
        )
        return

    employee_id = checklist["employee_id"]
    documents = checklist.get("documents", [])
    validated_count = sum(1 for d in documents if d["status"] == "validated")
    sent_to_hr = checklist.get("sent_to_hr", False)
    status_line = "Sent to HR &#10003;" if sent_to_hr else f"{validated_count} of {len(documents)} documents validated"
    st.markdown(
        '<div style="font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.06em;'
        f'color:oklch(55% 0.012 250);padding:14px 0 0 0;">Checklist &mdash; {html.escape(employee_id)}</div>'
        f'<div style="font-size:12px;color:{"oklch(45% 0.13 155)" if sent_to_hr else "oklch(48% 0.012 250)"};'
        f'padding:2px 0 6px 0;{"font-weight:600;" if sent_to_hr else ""}">{status_line}</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for doc in documents:
        label = _DOC_TYPE_LABELS.get(doc["doc_type"], doc["doc_type"])
        rows.append(
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            'padding:7px 0;border-bottom:1px solid oklch(93% 0.006 250);">'
            f'<span style="font-size:13px;color:oklch(28% 0.015 255);">{label}</span>'
            f'{_status_badge_html(doc["status"])}'
            "</div>"
        )
    if rows:
        rows[-1] = rows[-1].replace("border-bottom:1px solid oklch(93% 0.006 250);", "")
    st.markdown(
        '<div style="display:flex;flex-direction:column;padding:2px 10px;border-radius:10px;'
        'background:oklch(99% 0.002 250);border:1px solid oklch(90% 0.006 250);">'
        + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def _submit_document(doc_type: str, upload_file) -> None:
    """Validates and posts a single document to POST /upload-doc, writing its
    result into last_upload_result[doc_type] so it lands on that document's
    own card, then reruns. Single code path for both cards' Submit buttons.

    Runs inside st.spinner(), which blocks and animates in place during the
    call -- extraction genuinely takes a few seconds (a real Gemini vision
    call), and without this the UI just looked frozen. This is a different
    mechanism from the chat composer's typing-indicator pattern: that one
    defers rendering to the NEXT script run because the reply needs to
    appear as a new message row after a rerun; here nothing needs to survive
    a rerun mid-request, so the simpler synchronous st.spinner is the right
    tool, not a duplicate of that pattern.

    The result banner IS deferred to the next run (session_state +
    st.rerun() after a successful submit) -- that part still needs it, so
    the banner survives the rerun a successful submission triggers, which is
    also what makes the checklist refresh without a manual reload."""
    employee_id = st.session_state.employee_id.strip()
    dob_text = st.session_state.upload_dob.strip()
    dob_valid = _is_valid_iso_date(dob_text)
    if not (employee_id and st.session_state.upload_full_name and dob_text and upload_file):
        st.error("Employee ID, full name, date of birth, and a file are all required.")
    elif not dob_valid:
        st.error("Date of birth must be a real date in YYYY-MM-DD format (e.g. 1990-01-01).")
    else:
        with st.spinner("Verifying document — this can take a few seconds…"):
            try:
                response = requests.post(
                    f"{st.session_state.api_url.rstrip('/')}/upload-doc",
                    data={
                        "employee_id": employee_id,
                        "doc_type": doc_type,
                        "full_name": st.session_state.upload_full_name,
                        "date_of_birth": dob_text,
                        "faculty_class": st.session_state.upload_faculty_class,
                    },
                    files={"file": (upload_file.name, upload_file.getvalue(), upload_file.type)},
                    timeout=90,
                )
                response.raise_for_status()
                st.session_state.last_upload_result[doc_type] = response.json()
            except requests.RequestException as exc:
                st.session_state.last_upload_result[doc_type] = {"error": str(exc)}
        st.rerun()


def _render_document_card(doc_type: str, status: str) -> None:
    """One document's dropzone + Submit + own result banner. The dropzone
    renders unconditionally, including for an already-validated document --
    re-upload after a rejected/needs_review outcome is a real path."""
    label = _DOC_TYPE_LABELS.get(doc_type, doc_type)
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 0 2px 0;">'
        f'<span style="font-size:13.5px;font-weight:600;color:oklch(24% 0.015 255);">{html.escape(label)}</span>'
        f'{_status_badge_html(status)}'
        "</div>",
        unsafe_allow_html=True,
    )
    upload_file = st.file_uploader(
        "File (JPEG/PNG)", type=["png", "jpg", "jpeg"], key=f"upload_file_{doc_type}",
        label_visibility="collapsed",
    )
    if upload_file is not None:
        st.image(upload_file, width=180)

    if st.button("Submit", key=f"upload_submit_{doc_type}", use_container_width=True):
        _submit_document(doc_type, upload_file)

    result = st.session_state.last_upload_result.get(doc_type)
    if result:
        if "error" in result:
            st.markdown(
                f'<div style="margin-top:8px;padding:10px 12px;border-radius:10px;'
                f'background:oklch(96% 0.03 25);border:1px solid oklch(87% 0.06 25);'
                f'color:oklch(45% 0.15 25);font-size:13px;">Upload failed: {html.escape(result["error"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            validation = result["validation"]
            text_color, bg_color, border_color = _status_badge_style(validation["outcome"])
            st.markdown(
                f'<div style="margin-top:8px;padding:10px 12px;border-radius:10px;'
                f'background:{bg_color};border:1px solid {border_color};color:{text_color};'
                f'font-size:13px;">{html.escape(validation["message"])}</div>',
                unsafe_allow_html=True,
            )


def _render_document_panel(checklist: dict | None) -> None:
    """Document verification flow (Component 14) -- posts to POST
    /upload-doc. Lives in the main chat column (not the sidebar, which
    holds the Employee ID field and the compact checklist) so it sits
    alongside the conversation rather than off to the side.

    One card per config.REQUIRED_ONBOARDING_DOCS entry (not two hardcoded
    blocks) -- a future third required doc type gets a card automatically.
    Identity fields are entered once, above the cards, and shared by both
    documents' submissions."""
    with st.container(key="uploader_section"):
        # Streamlit 1.45 has no key= param for st.expander (added later), so
        # there's no widget state to fall back on across reruns -- expanded=
        # is re-applied fresh on every rerun, full stop. Passing anything
        # other than a constant True here would re-collapse the panel the
        # moment an unrelated widget (e.g. the Employee ID input) triggers a
        # rerun. Always-open is the tradeoff until this project's Streamlit
        # pin moves past 1.47.
        with st.expander("Document verification", expanded=True):
            st.session_state.upload_full_name = st.text_input(
                "Full name (as printed on the document)",
                value=st.session_state.upload_full_name, key="upload_full_name_input",
                placeholder="e.g. REYES, MARIA SANTOS",
            )
            st.session_state.upload_dob = st.text_input(
                "Date of birth (YYYY-MM-DD)", value=st.session_state.upload_dob, key="upload_dob_input",
                placeholder="e.g. 1990-01-01",
            )
            # Drives the HR handoff email's per-class "still outstanding"
            # checklist (config.PREEMPLOYMENT_CHECKLIST) once both documents
            # validate — the three faculty classes carry different
            # requirement sets (CLAUDE.md's biggest corpus/process hazard).
            st.session_state.upload_faculty_class = st.selectbox(
                "Faculty class", config.AUDIENCE_ORDER,
                index=config.AUDIENCE_ORDER.index(st.session_state.upload_faculty_class),
                format_func=lambda slug: config.AUDIENCE_LABELS.get(slug, slug),
                key="upload_faculty_class_input",
            )

            for doc_type in config.REQUIRED_ONBOARDING_DOCS:
                st.divider()
                _render_document_card(doc_type, _doc_status(checklist, doc_type))


def _render_sidebar(accent: str, dev_mode: bool) -> None:
    with st.sidebar:
        st.markdown(
            f'''<div style="display:flex;flex-direction:column;">
  <div style="display:flex;align-items:center;gap:8px;padding:6px 0 18px 0;">
    <div style="width:22px;height:22px;border-radius:6px;background:{accent};flex-shrink:0;"></div>
    <div style="font-size:14px;font-weight:600;letter-spacing:0.01em;color:oklch(20% 0.015 255);">{config.ASSISTANT_NAME}</div>
  </div>
  <div style="font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.06em;color:oklch(55% 0.012 250);padding:4px 0 8px 0;">Recent</div>
  <div style="display:flex;flex-direction:column;gap:1px;padding:9px 8px;border-radius:8px;background:oklch(92% 0.012 250);">
    <div style="font-size:13.5px;font-weight:500;color:oklch(22% 0.015 255);">Current chat</div>
    <div style="font-size:11.5px;color:oklch(56% 0.012 250);">Today</div>
  </div>
</div>''',
            unsafe_allow_html=True,
        )
        # DOCUMENTS is always visible so the CV/OCR track doesn't depend on
        # the chat router correctly classifying a document-upload intent
        # (Intent.DOCUMENT_UPLOAD/DOCUMENT_STATUS still opens it too, via
        # show_upload_flow -- see _fetch_pending_response) -- but the
        # Employee ID field only appears once the user actually asks for it,
        # preserving the original "no HR form for Manual-only questions"
        # intent. Persists for the rest of the session once opened.
        st.markdown(
            '<div style="font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.06em;'
            'color:oklch(55% 0.012 250);padding:14px 0 6px 0;border-top:1px solid oklch(90% 0.006 250);'
            'margin-top:10px;">Documents</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.show_upload_flow:
            st.session_state.employee_id = st.text_input(
                "Employee ID", value=st.session_state.employee_id, key="employee_id_input",
                placeholder="e.g. EMP-04821",
            )
        else:
            st.button(
                "Verify a document", key="open_doc_flow_btn",
                on_click=_open_document_flow, use_container_width=True,
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
                f'<div style="font-size:15px;font-weight:600;color:oklch(24% 0.015 255);">{config.ASSISTANT_NAME}</div>'
                '<div style="font-size:12px;color:oklch(52% 0.012 250);">Grounded answers, with citations</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        with col3:
            with st.container(key="new_chat_wrap"):
                st.button("+  New chat", key="new_chat_btn", on_click=_start_new_chat)


def _render_actions(actions: list[dict]) -> None:
    for action in actions:
        label = html.escape(action.get("label", ""))
        status = action.get("status", "completed")
        if status == "pending":
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


def _queue_message(prompt: str) -> None:
    """Appends the user's turn and queues the API call for the next script
    run. Shared by the composer and the quick-start chips so there's exactly
    one code path that talks to the API. Split from the actual request (see
    `_fetch_pending_response`) so a typing indicator can render in-flow
    *before* the blocking network call, instead of a spinner appearing
    outside the message list."""
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
    st.session_state.pending_request = {"message": prompt}
    st.session_state.awaiting_response = True


def _fetch_pending_response() -> None:
    """Performs the queued API call and appends the assistant's reply. Must
    only be called after the typing indicator has already been rendered."""
    pending = st.session_state.pending_request or {}
    payload: dict = {
        "session_id": st.session_state.session_id,
        "message": pending.get("message", ""),
        "employee_id": st.session_state.employee_id or None,
    }

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

    actions = data.get("actions", [])
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["reply"],
            "citations": data.get("citations", []),
            "sources": data.get("sources", []),
            "web_citations": data.get("web_citations", []),
            "actions": actions,
            "token_usage": data.get("token_usage", {}),
        }
    )
    # Reveals the Employee ID field, checklist, and uploader once the
    # conversation actually asks for document verification -- persists for
    # the rest of the session (see _init_state's note), so this only ever
    # flips False -> True here, never back.
    if any(a.get("type") == "unlock_document_flow" for a in actions):
        st.session_state.show_upload_flow = True
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


def _render_message(i: int, msg: dict, accent: str) -> None:
    with st.container(key=f"msg_{i}"):
        if msg["role"] == "assistant":
            # Avatar + Markdown body as two children of a flex-row container (styled
            # in the global CSS). The body goes through st.markdown so bold, lists,
            # and tables render; it is NOT html.escape'd here — the model's answer is
            # trusted Markdown, and Streamlit sanitizes it (no raw HTML passthrough).
            with st.container(key=f"msg_row_{i}"):
                st.markdown(
                    f'<div style="width:26px;height:26px;border-radius:7px;background:{accent};'
                    'flex-shrink:0;margin-top:2px;display:flex;align-items:center;justify-content:center;">'
                    '<div style="width:8px;height:8px;border-radius:2px;background:oklch(99% 0 0);"></div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(msg["content"])
            _render_actions(msg.get("actions") or [])
            _render_pills_and_panels(i, msg, accent)
        else:
            # User messages stay escaped — a user's text must never be interpreted
            # as Markdown or HTML.
            content = html.escape(msg["content"])
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
        for i, message in enumerate(st.session_state.messages):
            _render_message(i, message, accent)


st.set_page_config(page_title=config.ASSISTANT_NAME, page_icon="💬", layout="wide")
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
# Fetched once here, after the sidebar's Employee ID input has run, and
# shared by both the sidebar's compact checklist and the main-column
# document cards below -- avoids fetching GET /onboarding-status twice per
# script run.
_checklist = _fetch_checklist()
with st.sidebar:
    _render_sidebar_checklist(_checklist)
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
if st.session_state.show_upload_flow:
    _render_document_panel(_checklist)

_prompt = st.chat_input("Ask about faculty onboarding or the Faculty Manual…")
if _prompt:
    _queue_message(_prompt)
    st.rerun()