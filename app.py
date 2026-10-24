"""VoxTrace local, investigator-support WAV review interface."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

import pandas as pd
import streamlit as st

from config import (
    EXPECTED_ID2LABEL,
    MAX_UPLOAD_MB,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    LONG_RECORDING_SECONDS,
    PROVISIONAL_THRESHOLDS,
)
from engine import AudioValidationError, ModelVerificationError, analyze_audio, dependency_versions, load_audio, load_model
from receipt import build_receipt, strict_json_dumps, verify_sha256
from receipt import sha256_bytes
from session_state import bind_state_to_upload, clear_upload_state
from signals import rank_candidates
from ui_text import (
    COVERAGE_DESCRIPTION,
    LOW_SCORE_NOTICE,
    MODEL_LIMITATIONS,
    QUEUE_TITLE,
    SCORE_DESCRIPTION,
    SCOPE_BANNER,
)


st.set_page_config(page_title="VoxTrace", page_icon="◌", layout="wide")

def read_verification_record() -> dict:
    record_path = Path(__file__).with_name("MODEL_VERIFICATION.json")
    try:
        return json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "missing", "model_loaded": False}


@st.cache_resource(show_spinner=False, max_entries=1)
def cached_model_bundle():
    return load_model()


def clear_fingerprint_result() -> None:
    st.session_state.pop("voxtrace_fingerprint_result", None)


def verification_record_is_ready(record: dict) -> bool:
    try:
        mapping = {int(key): value for key, value in record.get("config", {}).get("id2label", {}).items()}
    except (TypeError, ValueError):
        return False
    return (
        record.get("status") == "verified"
        and record.get("model_loaded") is True
        and record.get("loaded_revision") == MODEL_REVISION
        and mapping == EXPECTED_ID2LABEL
        and record.get("config", {}).get("num_labels") == len(EXPECTED_ID2LABEL)
        and record.get("classifier_head", {}).get("observed_output_dimension") == len(EXPECTED_ID2LABEL)
        and record.get("synthetic_forward_pass", {}).get("status") == "passed"
    )


def window_heading(window: dict) -> str:
    flags = ", flagged" if window.get("quality_flags") else ""
    return (
        f"{window['start_seconds']:.2f}s–{window['end_seconds']:.2f}s"
        f" · {window['status']}{flags} · window {window['index'] + 1}"
    )


def show_window(window: dict, audio_data) -> None:
    st.write(
        f"**Window:** {window['start_seconds']:.2f}–{window['end_seconds']:.2f} seconds  "
        f"· **Status:** {window['status']}"
    )
    if window.get("reason"):
        st.write(f"**Reason:** {window['reason']}")
    if window.get("quality_flags"):
        st.write("**Quality flags:** " + ", ".join(window["quality_flags"]))
    else:
        st.write("**Quality flags:** none recorded")
    metrics = window.get("metrics", {})
    if metrics:
        st.caption(
            "Acoustic screening indicators: "
            f"RMS {metrics.get('rms_energy', 0.0):.5f} · "
            f"peak {metrics.get('peak_amplitude', 0.0):.4f} · "
            f"VAD activity {metrics.get('vad_activity_ratio')}"
        )
    if window.get("status") == "classified":
        score_frame = pd.DataFrame(window["scores"])
        st.dataframe(
            score_frame,
            hide_index=True,
            width="stretch",
            alt="All verified raw model scores for this classified window",
        )
    start = int(round(window["start_seconds"] * audio_data.sample_rate))
    end = min(
        int(round(window["end_seconds"] * audio_data.sample_rate)),
        len(audio_data.samples),
    )
    snippet = audio_data.samples[start:end]
    if snippet.size:
        st.audio(snippet, sample_rate=audio_data.sample_rate)


st.title("VoxTrace")
st.caption("Cybersecurity · audio review prototype")
st.info(SCOPE_BANNER, icon=":material/info:")

uploaded_file = st.file_uploader(
    "Upload a recorded WAV",
    type=["wav"],
    max_upload_size=MAX_UPLOAD_MB,
    help=f"Local WAV analysis only. Maximum upload size: {MAX_UPLOAD_MB} MB.",
    key="voxtrace_upload",
)

if uploaded_file is None:
    clear_upload_state(st.session_state)
    st.info("Upload a WAV recording to inspect its metadata and begin a local review.")
    st.stop()

file_bytes = uploaded_file.getvalue()
current_hash = sha256_bytes(file_bytes)
if bind_state_to_upload(st.session_state, current_hash):
    st.session_state["voxtrace_audio_error"] = None

try:
    if st.session_state.get("voxtrace_loaded_hash") != current_hash:
        audio_data = load_audio(file_bytes)
        st.session_state["voxtrace_audio_data"] = audio_data
        st.session_state["voxtrace_loaded_hash"] = current_hash
        st.session_state["voxtrace_audio_error"] = None
    audio_data = st.session_state["voxtrace_audio_data"]
except AudioValidationError as exc:
    audio_data = None
    st.session_state["voxtrace_audio_error"] = str(exc)

st.header("1 · Recording and WAV metadata")
left, right = st.columns([1.45, 1])
with left:
    st.audio(file_bytes, format="audio/wav")
    st.caption("Original uploaded recording. Audio is processed locally by this app.")
with right:
    st.write(f"**SHA-256 of uploaded bytes:** `{current_hash}`")
    st.caption(
        "A matching fingerprint demonstrates byte equality with a supplied reference. "
        "It does not establish recording authenticity or speaker identity."
    )
    if audio_data is not None:
        st.dataframe(
            pd.DataFrame([audio_data.original_wav_metadata]),
            hide_index=True,
            width="stretch",
            alt="Original WAV format, subtype, channels, sample rate, frame count, and duration",
        )
    elif st.session_state.get("voxtrace_audio_error"):
        st.error(st.session_state["voxtrace_audio_error"])

if audio_data is not None and audio_data.original_wav_metadata["duration_seconds"] >= LONG_RECORDING_SECONDS:
    st.warning("This recording is long enough that CPU analysis may take substantial time.")

st.header("2 · Pinned model identity")
verification_record = read_verification_record()
model_ready = verification_record_is_ready(verification_record)
model_left, model_right = st.columns([1, 1])
with model_left:
    st.write(f"**Checkpoint:** `{MODEL_ID}`")
    st.write(f"**Pinned revision:** `{MODEL_REVISION}`")
    st.write(f"**Verified loaded revision:** `{verification_record.get('loaded_revision', 'unavailable')}`")
    st.write(f"**Verification record:** {verification_record.get('status', 'missing')}")
    st.write(f"**Feature-extractor sampling rate:** {verification_record.get('feature_extractor', {}).get('sampling_rate_hz', 'unavailable')} Hz")
with model_right:
    if model_ready:
        mapping_frame = pd.DataFrame([
            {"Class ID": index, "Raw checkpoint label": label}
            for index, label in sorted(EXPECTED_ID2LABEL.items())
        ])
        st.dataframe(
            mapping_frame,
            hide_index=True,
            width="stretch",
            alt="Verified class IDs and raw checkpoint labels",
        )
    else:
        st.warning(
            "Model analysis is disabled because the local verification record does not "
            "confirm this exact pinned revision, label map, and output dimension. "
            "Run `python scripts\\verify_model.py` in the project environment."
        )
st.caption(MODEL_LIMITATIONS)

if audio_data is not None:
    analyze_clicked = st.button(
        "Analyze recording",
        type="primary",
        disabled=not model_ready,
    )
    if analyze_clicked:
        progress = st.progress(0, text="Loading and rechecking the pinned local model…")
        try:
            bundle = cached_model_bundle()

            def update_progress(done: int, total: int) -> None:
                progress.progress(done / max(total, 1), text=f"Classifying window {done} of {total}…")

            windows, coverage, stats = analyze_audio(
                audio_data,
                bundle,
                progress_callback=update_progress,
            )
            model_receipt = {
                "model_id": MODEL_ID,
                "requested_revision": MODEL_REVISION,
                "loaded_revision": bundle.revision,
                "verification_status": "verified",
                "raw_id2label": {
                    str(index): label for index, label in EXPECTED_ID2LABEL.items()
                },
                "label2id": {label: index for index, label in EXPECTED_ID2LABEL.items()},
                "num_labels": bundle.output_dimension,
                "output_dimension": bundle.output_dimension,
                "feature_extractor_sampling_rate_hz": bundle.sample_rate,
                "classifier_head_linear_modules": bundle.verification["classifier_head_linear_modules"],
                "safetensors_requested": True,
                "cached_weight_sha256": verification_record.get("artifacts", {}).get("model_safetensors_sha256"),
            }
            receipt = build_receipt(
                input_bytes=file_bytes,
                original_wav_metadata=audio_data.original_wav_metadata,
                audio_transform=audio_data.transform,
                model=model_receipt,
                windows=windows,
                coverage=coverage,
                class_statistics=stats,
                dependency_versions=dependency_versions(),
                provisional_thresholds=PROVISIONAL_THRESHOLDS,
            )
            strict_json_dumps(receipt)
            st.session_state["voxtrace_results"] = windows
            st.session_state["voxtrace_coverage"] = coverage
            st.session_state["voxtrace_class_statistics"] = stats
            st.session_state["voxtrace_receipt"] = receipt
            st.session_state["voxtrace_analysis_error"] = None
        except (ModelVerificationError, AudioValidationError, Exception) as exc:
            st.session_state.pop("voxtrace_results", None)
            st.session_state.pop("voxtrace_coverage", None)
            st.session_state.pop("voxtrace_class_statistics", None)
            st.session_state.pop("voxtrace_receipt", None)
            st.session_state["voxtrace_analysis_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            progress.empty()

if st.session_state.get("voxtrace_analysis_error"):
    st.error(
        "Analysis could not be completed safely: "
        + st.session_state["voxtrace_analysis_error"]
    )

if (
    st.session_state.get("voxtrace_loaded_hash") == current_hash
    and st.session_state.get("voxtrace_results") is not None
):
    windows = st.session_state["voxtrace_results"]
    coverage = st.session_state["voxtrace_coverage"]
    stats = st.session_state["voxtrace_class_statistics"]

    st.header("3 · Processing coverage")
    state = coverage["coverage_state"]
    if state == "NO_ANALYZED_AUDIO":
        st.error("No windows were successfully classified. Review the window log for failures and skipped intervals.")
    elif state == "INSUFFICIENT_COVERAGE":
        st.warning(
            "Insufficient processing coverage for file-level class summaries. "
            "Inspect individual classified windows; this warning remains relevant to all results."
        )
    elif state == "PARTIAL_COVERAGE":
        st.warning(
            "Partial coverage. Short recordings receive this status even when every usable sample was classified. "
            "File-level class summaries are suppressed."
        )
    else:
        st.info("Adequate provisional processing coverage. This state does not validate emotion predictions or establish safety.")

    st.caption(COVERAGE_DESCRIPTION)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Recording", f"{coverage['recording_duration_seconds']:.2f} s")
    metric_cols[1].metric("Classified union", f"{coverage['classified_all_seconds']:.2f} s")
    metric_cols[2].metric("Unanalyzed", f"{coverage['unanalyzed_seconds']:.2f} s")
    metric_cols[3].metric("Analyzed time", f"{coverage['analyzed_time_fraction']:.1%}")
    metric_cols[4].metric("Unflagged time", f"{coverage['unflagged_time_fraction']:.1%}")
    count_cols = st.columns(5)
    count_cols[0].metric("Total windows", coverage["total_windows"])
    count_cols[1].metric("Classified", coverage["successfully_classified_windows"])
    count_cols[2].metric("Flagged", coverage["flagged_classified_windows"])
    count_cols[3].metric("Skipped", coverage["skipped_windows"])
    count_cols[4].metric("Failed", coverage["failed_windows"])
    time_cols = st.columns(3)
    time_cols[0].metric(
        "Skipped window union", f"{coverage['skipped_window_interval_union_seconds']:.2f} s"
    )
    time_cols[1].metric(
        "Failed window union", f"{coverage['failed_window_interval_union_seconds']:.2f} s"
    )
    time_cols[2].metric("Unprocessed time", f"{coverage['unprocessed_seconds']:.2f} s")
    st.caption(
        "Skipped and failed interval unions can overlap classified or each other because windows overlap; "
        "these outcome figures are not additive. Unprocessed time has no completed window outcome."
    )

    st.header("4 · Raw class summaries")
    st.info(LOW_SCORE_NOTICE)
    st.caption(SCORE_DESCRIPTION)
    if stats is None:
        st.write(
            "Per-class file summaries are unavailable for this coverage state. "
            "Classified individual windows remain available below."
        )
    else:
        for group_name, title in (
            ("all_classified_windows", "All successfully classified windows"),
            ("unflagged_classified_windows", "Classified windows without quality flags"),
        ):
            with st.expander(title, expanded=(group_name == "all_classified_windows")):
                summary_rows = []
                for label, values in stats[group_name].items():
                    if values is not None:
                        summary_rows.append({
                            "Raw class": label,
                            "Mean": values["mean"],
                            "Median": values["median"],
                            "Maximum": values["maximum"],
                            "Windows": values["window_count"],
                        })
                st.dataframe(
                    pd.DataFrame(summary_rows),
                    hide_index=True,
                    width="stretch",
                    alt=f"Raw class summary for {title.lower()}",
                )
                st.caption(
                    "Descriptive summaries over overlapping audio windows. These are not independent samples, "
                    "calibrated emotion probabilities, or evidence of a caller's emotional state."
                )

    st.header("5 · Listen-first candidate review")
    st.subheader(QUEUE_TITLE)
    available_labels = [EXPECTED_ID2LABEL[index] for index in sorted(EXPECTED_ID2LABEL)]
    selectors = list(available_labels)
    if {"angry", "fearful"}.issubset(set(available_labels)):
        selectors.insert(0, "max(angry, fearful)")
    default_selector = "max(angry, fearful)" if "max(angry, fearful)" in selectors else selectors[0]
    selected_selector = st.selectbox(
        "Raw class used to order windows",
        selectors,
        index=selectors.index(default_selector),
        help="Scores order windows for listening only. No score threshold indicates an emotion or threat verdict.",
    )

    classified = [window for window in windows if window["status"] == "classified"]
    unflagged = [window for window in classified if not window["quality_flags"]]
    flagged = [window for window in classified if window["quality_flags"]]
    unflagged_queue = rank_candidates(unflagged, selected_selector, available_labels)
    if not unflagged_queue:
        st.info(
            "No unflagged windows were classified. Review the separate flagged-window queue below; "
            "the absence of unflagged candidates says nothing about safety."
        )
    else:
        for candidate in unflagged_queue:
            with st.expander(
                f"{window_heading(candidate)} · {candidate['selected_class']} raw score {candidate['selected_score']:.4f}"
            ):
                st.write(
                    f"**Selected raw output:** {candidate['selected_class']} · "
                    f"{candidate['selected_score']:.6f}"
                )
                show_window(candidate, audio_data)

    with st.expander(f"Flagged classified windows · {len(flagged)} available", expanded=False):
        st.caption(
            "Flags are screening heuristics, not proof that speech is absent. Flagged audio remains available for listening."
        )
        flagged_queue = rank_candidates(flagged, selected_selector, available_labels)
        if not flagged_queue:
            st.write("No successfully classified windows with quality flags.")
        for candidate in flagged_queue:
            with st.container(border=True):
                st.subheader(
                    f"{window_heading(candidate)} · {candidate['selected_class']} raw score {candidate['selected_score']:.4f}"
                )
                st.write(
                    f"**Selected raw output:** {candidate['selected_class']} · "
                    f"{candidate['selected_score']:.6f}"
                )
                show_window(candidate, audio_data)

    st.header("6 · Complete window log")
    log_rows = []
    for window in windows:
        row = {
            "Window": window["index"] + 1,
            "Start (s)": window["start_seconds"],
            "End (s)": window["end_seconds"],
            "Status": window["status"],
            "Quality flags": ", ".join(window["quality_flags"]),
            "Reason": window["reason"],
        }
        row.update(window.get("scores_by_label", {}))
        log_rows.append(row)
    st.dataframe(
        pd.DataFrame(log_rows),
        hide_index=True,
        width="stretch",
        alt="Complete per-window statuses, screening flags, and raw class scores",
    )
    with st.expander("Window details and playback", expanded=False):
        for window in windows:
            with st.container(border=True):
                st.subheader(window_heading(window))
                show_window(window, audio_data)

    st.header("7 · Reproducibility receipt")
    receipt = st.session_state["voxtrace_receipt"]
    receipt_json = strict_json_dumps(receipt)
    st.download_button(
        "Download JSON receipt",
        data=receipt_json,
        file_name="voxtrace_analysis_receipt.json",
        mime="application/json",
        key=f"voxtrace_receipt_{current_hash[:12]}",
    )
    st.write("**Receipt input SHA-256:** " + receipt["input"]["sha256"])
    st.caption("The receipt excludes raw audio arrays and uploaded audio bytes.")

    st.subheader("Verify a previously recorded fingerprint")
    expected_fingerprint = st.text_input(
        "Expected SHA-256 fingerprint",
        value="",
        key="voxtrace_fingerprint_input",
        help="Paste a fingerprint recorded separately. The field starts blank.",
        on_change=clear_fingerprint_result,
    )
    if st.button("Verify uploaded bytes", key="voxtrace_verify_fingerprint"):
        matches = verify_sha256(file_bytes, expected_fingerprint)
        st.session_state["voxtrace_fingerprint_result"] = (
            "match" if matches else "mismatch_or_invalid"
        )
    fingerprint_result = st.session_state.get("voxtrace_fingerprint_result")
    if fingerprint_result == "match":
        st.info("The uploaded bytes match the supplied SHA-256 fingerprint.")
    elif fingerprint_result == "mismatch_or_invalid":
        st.warning("The supplied value is invalid or the uploaded bytes do not match it.")

st.divider()
st.caption(
    "VoxTrace is an experimental human-review aid. Its outputs are not definitive evidence and must not determine dispatch, "
    "operational triage, guilt, fraud, deception, danger, urgency, or clinical status."
)
