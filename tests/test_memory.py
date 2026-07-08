import pytest

from src import config
from src.memory import persistent, session
from src.schemas import SessionSummary


@pytest.fixture(autouse=True)
def isolated_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")


class FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed


class FakeModels:
    def __init__(self, parsed):
        self._parsed = parsed

    def generate_content(self, model, contents, config):
        return FakeResponse(self._parsed)


class FakeClient:
    def __init__(self, parsed):
        self.models = FakeModels(parsed)


# --- session.py: short-term history, trimming -------------------------------

def test_append_and_get_history_roundtrip():
    session.append_turn("s1", "user", "hello")
    session.append_turn("s1", "assistant", "hi there")
    assert session.get_history("s1") == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_get_history_is_scoped_per_session():
    session.append_turn("s1", "user", "message in s1")
    session.append_turn("s2", "user", "message in s2")
    assert session.get_history("s1") == [{"role": "user", "content": "message in s1"}]
    assert session.get_history("s2") == [{"role": "user", "content": "message in s2"}]


def test_get_history_trims_to_limit_keeping_most_recent():
    for i in range(5):
        session.append_turn("s1", "user", f"turn {i}")
    trimmed = session.get_history("s1", limit=2)
    assert trimmed == [
        {"role": "user", "content": "turn 3"},
        {"role": "user", "content": "turn 4"},
    ]


def test_get_full_history_is_never_trimmed():
    for i in range(5):
        session.append_turn("s1", "user", f"turn {i}")
    full = session.get_full_history("s1")
    assert len(full) == 5
    assert full[0]["turn_index"] == 0
    assert full[-1]["content"] == "turn 4"


def test_count_turns():
    assert session.count_turns("s1") == 0
    session.append_turn("s1", "user", "hi")
    assert session.count_turns("s1") == 1


# --- persistent.py: summary trigger on overflow -----------------------------

def _fill_turns(session_id: str, n: int) -> None:
    for i in range(n):
        session.append_turn(session_id, "user" if i % 2 == 0 else "assistant", f"turn {i}")


def test_maybe_update_summary_noop_under_trim_window():
    _fill_turns("s1", config.MEMORY_TRIM_TURNS)  # exactly at the window, no overflow yet
    client = FakeClient(SessionSummary(summary_text="should not be called"))
    persistent.maybe_update_summary("s1", client=client)
    assert persistent.get_summary("s1") is None


def test_maybe_update_summary_noop_below_batch_size():
    # overflow exists but hasn't reached a full batch yet
    _fill_turns("s1", config.MEMORY_TRIM_TURNS + config.MEMORY_SUMMARY_BATCH_SIZE - 1)
    client = FakeClient(SessionSummary(summary_text="should not be called"))
    persistent.maybe_update_summary("s1", client=client)
    assert persistent.get_summary("s1") is None


def test_maybe_update_summary_fires_once_batch_overflows():
    _fill_turns("s1", config.MEMORY_TRIM_TURNS + config.MEMORY_SUMMARY_BATCH_SIZE)
    client = FakeClient(SessionSummary(summary_text="employee asked about leave policy"))
    persistent.maybe_update_summary("s1", employee_id="emp-1", client=client)
    assert persistent.get_summary("s1") == "employee asked about leave policy"


def test_maybe_update_summary_extends_rather_than_restarts():
    _fill_turns("s1", config.MEMORY_TRIM_TURNS + config.MEMORY_SUMMARY_BATCH_SIZE)
    first_client = FakeClient(SessionSummary(summary_text="first summary"))
    persistent.maybe_update_summary("s1", client=first_client)
    assert persistent.get_summary("s1") == "first summary"

    # not enough new overflow yet for a second call to fire
    _fill_turns("s1", config.MEMORY_SUMMARY_BATCH_SIZE - 1)
    second_client = FakeClient(SessionSummary(summary_text="should not overwrite"))
    persistent.maybe_update_summary("s1", client=second_client)
    assert persistent.get_summary("s1") == "first summary"

    # one more turn crosses the next batch threshold
    _fill_turns("s1", 1)
    third_client = FakeClient(SessionSummary(summary_text="extended summary"))
    persistent.maybe_update_summary("s1", client=third_client)
    assert persistent.get_summary("s1") == "extended summary"


def test_maybe_update_summary_fails_closed_on_unparseable_response():
    _fill_turns("s1", config.MEMORY_TRIM_TURNS + config.MEMORY_SUMMARY_BATCH_SIZE)
    client = FakeClient(None)  # simulates response.parsed is None
    persistent.maybe_update_summary("s1", client=client)
    # nothing existing to fall back to yet, but must not raise
    assert persistent.get_summary("s1") == "(no prior summary)"


# --- persistent.py: cross-session recall + get_context -----------------------

def test_get_context_fresh_session_no_employee_returns_empty():
    assert persistent.get_context("brand-new-session") == []


def test_get_context_seeds_from_employees_latest_summary_on_new_session():
    _fill_turns("old-session", config.MEMORY_TRIM_TURNS + config.MEMORY_SUMMARY_BATCH_SIZE)
    client = FakeClient(SessionSummary(summary_text="discussed maternity leave eligibility"))
    persistent.maybe_update_summary("old-session", employee_id="emp-42", client=client)

    context = persistent.get_context("new-session", employee_id="emp-42")
    assert len(context) == 1
    assert "discussed maternity leave eligibility" in context[0]["content"]


def test_get_context_does_not_seed_for_unknown_employee():
    context = persistent.get_context("new-session", employee_id="unknown-employee")
    assert context == []


def test_get_context_combines_own_summary_with_trimmed_recent_turns():
    _fill_turns("s1", config.MEMORY_TRIM_TURNS + config.MEMORY_SUMMARY_BATCH_SIZE)
    client = FakeClient(SessionSummary(summary_text="earlier: asked about payroll dates"))
    persistent.maybe_update_summary("s1", client=client)

    context = persistent.get_context("s1")
    assert "earlier: asked about payroll dates" in context[0]["content"]
    assert len(context) == 1 + config.MEMORY_TRIM_TURNS
