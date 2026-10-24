from __future__ import annotations

from pathlib import Path

from session_state import bind_state_to_upload, clear_upload_state
from ui_text import LOW_SCORE_NOTICE, MODEL_LIMITATIONS, QUEUE_TITLE, SCOPE_BANNER


def test_required_scope_and_raw_label_limitations_are_present():
    assert "does not determine danger" in SCOPE_BANNER
    assert "deception" in SCOPE_BANNER and "dispatch" in SCOPE_BANNER
    assert "Nervousness and distress are not modelled as independent classes" in MODEL_LIMITATIONS
    assert "`calm` is not relabelled `neutral`" in MODEL_LIMITATIONS
    assert "closed set" in MODEL_LIMITATIONS
    assert "does not mean the call is safe, calm" in LOW_SCORE_NOTICE
    assert "Unanalyzed audio has not been assessed." in LOW_SCORE_NOTICE
    assert QUEUE_TITLE == "Listen-First Candidate Windows — Sorted by Raw Model Output, Not an Emotion or Threat Verdict"


def test_forbidden_advisory_language_and_proxy_formulas_are_absent_from_ui_copy():
    forbidden = (
        "Limited detected review signal",
        "Elevated review signal",
        "nervousness = max(fear, surprise)",
        "distress = max(fear, sadness)",
        "urgency score",
        "danger score",
    )
    content = "\n".join((SCOPE_BANNER, MODEL_LIMITATIONS, LOW_SCORE_NOTICE, QUEUE_TITLE)).lower()
    for phrase in forbidden:
        assert phrase.lower() not in content


def test_replacing_upload_hash_clears_all_prior_results_and_receipts():
    state = {
        "voxtrace_upload_sha256": "old",
        "voxtrace_results": [{"status": "classified"}],
        "voxtrace_receipt": {"input": {"sha256": "old"}},
        "voxtrace_audio_data": object(),
        "voxtrace_fingerprint_result": "match",
    }
    assert bind_state_to_upload(state, "new")
    assert state["voxtrace_upload_sha256"] == "new"
    assert "voxtrace_results" not in state
    assert "voxtrace_receipt" not in state
    assert "voxtrace_audio_data" not in state
    assert "voxtrace_fingerprint_result" not in state


def test_removing_upload_clears_results_receipt_and_source_hash():
    state = {
        "voxtrace_upload_sha256": "current",
        "voxtrace_results": [1],
        "voxtrace_receipt": {"input": {"sha256": "current"}},
        "voxtrace_fingerprint_input": "some previous value",
    }
    clear_upload_state(state)
    assert state == {}


def test_app_binds_visible_results_to_current_upload_hash():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "clear_upload_state(st.session_state)" in source
    assert "bind_state_to_upload(st.session_state, current_hash)" in source
    assert 'st.session_state.get("voxtrace_loaded_hash") == current_hash' in source
