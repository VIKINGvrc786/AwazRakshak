from __future__ import annotations

from pathlib import Path

from config import TONAL_DOMINANT_POWER_FRACTION, TONAL_SPECTRAL_FLATNESS_MAX
from session_state import bind_state_to_upload, clear_upload_state
from ui_text import (
    LOW_SCORE_NOTICE,
    MODEL_LIMITATIONS,
    QUEUE_TITLE,
    SCOPE_BANNER,
    TONAL_FLAG_DESCRIPTION,
    format_measurement,
    format_raw_score,
    safe_analysis_error_message,
    summarize_top_outputs,
)


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


def test_tonal_flag_explains_both_measured_triggers_and_its_limit():
    assert (
        f"spectral flatness is ≤ {TONAL_SPECTRAL_FLATNESS_MAX:.3f}"
        in TONAL_FLAG_DESCRIPTION
    )
    assert (
        "dominant non-DC FFT-bin power share is ≥ "
        f"{TONAL_DOMINANT_POWER_FRACTION:.0%}" in TONAL_FLAG_DESCRIPTION
    )
    assert "harmonic or voiced-like audio can also trigger it" in TONAL_FLAG_DESCRIPTION
    assert "not a speech-presence or speech-absence test" in TONAL_FLAG_DESCRIPTION
    assert "provisional engineering cutoffs" in TONAL_FLAG_DESCRIPTION

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "st.caption(TONAL_FLAG_DESCRIPTION)" in source
    assert "metrics.get('spectral_flatness'" in source
    assert "metrics.get('dominant_peak_fraction'" in source


def test_missing_acoustic_measurements_are_not_displayed_as_zero():
    assert format_measurement(None, ".5f") == "unavailable"
    assert format_measurement(0.0, ".5f") == "0.00000"


def test_raw_score_summary_uses_stable_order_and_discloses_display_precision_ties():
    scores = [
        {"label": "angry", "score": 0.50004},
        {"label": "calm", "score": 0.50003},
        {"label": "fearful", "score": 0.2},
    ]
    top, runner_up, tied_labels = summarize_top_outputs(scores)

    assert top["label"] == "angry"
    assert runner_up["label"] == "calm"
    assert tied_labels == ("angry", "calm")
    assert format_raw_score(top["score"]) == format_raw_score(runner_up["score"]) == "0.5000"


def test_analysis_errors_are_actionable_without_raw_exception_text():
    class ModelVerificationError(Exception):
        pass

    raw_error = RuntimeError("C:\\private\\cache\\secret-token=123")
    assert "local model cache" in safe_analysis_error_message(raw_error)
    assert "secret-token" not in safe_analysis_error_message(raw_error)
    assert "verify_model.py" in safe_analysis_error_message(ModelVerificationError("bad pin"))


def test_interactive_window_review_is_wired_to_filtered_selection_and_local_playback():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert 'st.tabs(\n        ["Overview", "Window review", "Evidence & fingerprint"],' in source
    assert 'key="voxtrace_result_tabs"' in source
    assert 'on_change="rerun"' in source
    assert '"All windows", "Flagged classified", "Unflagged classified", "Skipped / failed"' in source
    assert "review_windows(" in source
    assert "select_review_window(visible_windows, selected_index)" in source
    assert "show_window(selected_window, audio_data, focal=True)" in source
    assert "TOP MODEL OUTPUT" in source
    assert "Runner-up:" in source
    assert "st.columns([2, 3], gap=\"large\")" in source
    assert 'st.download_button(' in source and '"Download JSON receipt"' in source
    assert 'file_name=f"voxtrace_analysis_receipt_{current_hash[:12]}.json"' in source
    assert 'key=f"voxtrace_receipt_{receipt_widget_suffix}"' in source


def test_replacing_upload_hash_clears_all_prior_results_and_receipts():
    state = {
        "voxtrace_upload_sha256": "old",
        "voxtrace_results": [{"status": "classified"}],
        "voxtrace_receipt": {"input": {"sha256": "old"}},
        "voxtrace_audio_data": object(),
        "voxtrace_fingerprint_result": "match",
        "voxtrace_window_filter": "Flagged",
        "voxtrace_window_order": "score",
        "voxtrace_window_selected": 9,
        "voxtrace_result_tabs": "Window review",
    }
    assert bind_state_to_upload(state, "new", "renamed.wav")
    assert state["voxtrace_upload_sha256"] == "new"
    assert state["voxtrace_upload_filename"] == "renamed.wav"
    assert "voxtrace_results" not in state
    assert "voxtrace_receipt" not in state
    assert "voxtrace_audio_data" not in state
    assert "voxtrace_fingerprint_result" not in state
    assert "voxtrace_window_filter" not in state
    assert "voxtrace_window_selected" not in state
    assert "voxtrace_result_tabs" not in state


def test_removing_upload_clears_results_receipt_and_source_hash():
    state = {
        "voxtrace_upload_sha256": "current",
        "voxtrace_results": [1],
        "voxtrace_receipt": {"input": {"sha256": "current"}},
        "voxtrace_fingerprint_input": "some previous value",
        "voxtrace_window_filter": "All windows",
    }
    clear_upload_state(state)
    assert state == {}


def test_same_bytes_with_a_different_filename_clears_filename_bound_analysis():
    state = {
        "voxtrace_upload_sha256": "same-hash",
        "voxtrace_upload_filename": "first.wav",
        "voxtrace_results": [1],
        "voxtrace_receipt": {"input": {"sha256": "same-hash", "filename": "first.wav"}},
    }

    assert bind_state_to_upload(state, "same-hash", "second.wav")
    assert state["voxtrace_upload_sha256"] == "same-hash"
    assert state["voxtrace_upload_filename"] == "second.wav"
    assert "voxtrace_results" not in state
    assert "voxtrace_receipt" not in state


def test_same_bytes_and_same_filename_preserves_current_analysis():
    state = {
        "voxtrace_upload_sha256": "same-hash",
        "voxtrace_upload_filename": "same.wav",
        "voxtrace_results": [1],
    }

    assert not bind_state_to_upload(state, "same-hash", "same.wav")
    assert state["voxtrace_results"] == [1]


def test_app_binds_visible_results_to_current_upload_hash():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "clear_upload_state(st.session_state)" in source
    assert "bind_state_to_upload(st.session_state, current_hash, current_filename)" in source
    assert 'st.session_state.get("voxtrace_loaded_hash") == current_hash' in source
    assert 'st.text(f"Uploaded filename: {current_filename}")' in source
