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
from engine import AudioValidationError, analyze_audio, dependency_versions, load_audio, load_model
from receipt import build_receipt, strict_json_dumps, verify_sha256
from receipt import sha256_bytes
from session_state import bind_state_to_upload, clear_upload_state
from signals import rank_candidates, review_windows, select_review_window
from ui_text import (
    COVERAGE_DESCRIPTION,
    LOW_SCORE_NOTICE,
    MODEL_LIMITATIONS,
    QUEUE_TITLE,
    SCORE_DESCRIPTION,
    SCOPE_BANNER,
    TONAL_FLAG_DESCRIPTION,
    format_raw_score,
    format_measurement,
    rank_raw_scores,
    safe_analysis_error_message,
    summarize_top_outputs,
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


def show_window(window: dict, audio_data, *, focal: bool = False) -> None:
    start_seconds = float(window["start_seconds"])
    end_seconds = float(window["end_seconds"])
    duration_seconds = end_seconds - start_seconds
    start_timestamp = f"{int(start_seconds // 60):02d}:{start_seconds % 60:05.2f}"
    end_timestamp = f"{int(end_seconds // 60):02d}:{end_seconds % 60:05.2f}"
    st.subheader(
        f"Window {window['index'] + 1} · {start_timestamp}–{end_timestamp} · {duration_seconds:.2f} s"
    )
    st.caption(f"Status: {window['status']}")
    if window.get("reason"):
        st.write(f"**Reason:** {window['reason']}")

    if focal and window.get("status") == "classified":
        top, runner_up, tied_labels = summarize_top_outputs(window["scores"])
        st.caption("TOP MODEL OUTPUT")
        headline_cols = st.columns([1.2, 1])
        headline_cols[0].markdown(f"## **{top['label'].upper()}**")
        headline_cols[1].metric("Raw classifier score", format_raw_score(top["score"]))
        if len(tied_labels) > 1:
            st.info(
                "Top model outputs are tied at four-decimal display precision: "
                + ", ".join(tied_labels)
                + "."
            )
        if runner_up is not None:
            st.caption(
                f"Runner-up: {runner_up['label']} · raw classifier score "
                f"{format_raw_score(runner_up['score'])}"
            )
        st.caption(
            "This is a closed-set model output for the selected audio window, "
            "not confirmation of a person's emotional state."
        )

    if window.get("quality_flags"):
        st.write("**Quality flags:** " + ", ".join(window["quality_flags"]))
        if "TONAL_SUSPECTED" in window["quality_flags"]:
            st.caption(TONAL_FLAG_DESCRIPTION)
    else:
        st.write("**Quality flags:** none recorded")
    start = int(round(start_seconds * audio_data.sample_rate))
    end = min(int(round(end_seconds * audio_data.sample_rate)), len(audio_data.samples))
    snippet = audio_data.samples[start:end]
    if snippet.size:
        st.audio(snippet, sample_rate=audio_data.sample_rate)
        st.caption("Playback is limited to this window's timestamped interval.")

    if window.get("status") == "classified":
        score_frame = pd.DataFrame([
            {
                "Raw class": row["label"],
                "Raw classifier score": format_raw_score(row["score"]),
            }
            for row in rank_raw_scores(window["scores"])
        ])
        st.dataframe(
            score_frame,
            hide_index=True,
            width="stretch",
            alt="All verified raw model scores for this classified window",
        )

    metrics = window.get("metrics", {})
    if metrics:
        def show_metrics() -> None:
            st.caption(
                "Acoustic screening indicators: "
                f"RMS {format_measurement(metrics.get('rms_energy'), '.5f')} · "
                f"peak {format_measurement(metrics.get('peak_amplitude'), '.4f')} · "
                f"spectral flatness {format_measurement(metrics.get('spectral_flatness'), '.5f')} · "
                "dominant FFT-bin power share "
                f"{format_measurement(metrics.get('dominant_peak_fraction'), '.1%')} · "
                f"VAD activity {format_measurement(metrics.get('vad_activity_ratio'), '.1%')}"
            )
        if focal:
            with st.expander("Acoustic screening details", expanded=False):
                show_metrics()
        else:
            show_metrics()


st.title("VoxTrace")
st.caption("Cybersecurity · local audio review workstation · experimental prototype")
st.info(SCOPE_BANNER, icon=":material/info:")
st.subheader("Recording workspace")
st.caption(
    f"Choose a WAV file (up to {MAX_UPLOAD_MB} MB). When run locally as documented, "
    "audio is processed on this machine. Review hosting and privacy implications "
    "before exposing the app to a network. Replacing the file clears its prior "
    "analysis and receipt."
)

uploaded_file = st.file_uploader(
    "Upload a recorded WAV",
    type=["wav"],
    max_upload_size=MAX_UPLOAD_MB,
    help=f"Local WAV analysis only. Maximum upload size: {MAX_UPLOAD_MB} MB.",
    key="voxtrace_upload",
)

if uploaded_file is None:
    clear_upload_state(st.session_state)
    st.info("No recording selected. Upload a WAV to inspect its metadata and begin a local review.")
    st.stop()

file_bytes = uploaded_file.getvalue()
current_hash = sha256_bytes(file_bytes)
current_filename = Path(uploaded_file.name.replace("\\", "/")).name
receipt_widget_suffix = sha256_bytes(
    f"{current_filename}\0{current_hash}".encode("utf-8")
)[:12]
if bind_state_to_upload(st.session_state, current_hash, current_filename):
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
    st.text(f"Uploaded filename: {current_filename}")
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
                input_filename=current_filename,
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
        except Exception as exc:
            st.session_state.pop("voxtrace_results", None)
            st.session_state.pop("voxtrace_coverage", None)
            st.session_state.pop("voxtrace_class_statistics", None)
            st.session_state.pop("voxtrace_receipt", None)
            st.session_state["voxtrace_analysis_error"] = safe_analysis_error_message(exc)
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
    if coverage["coverage_state"] == "ADEQUATE_PROVISIONAL_COVERAGE":
        st.success("Analysis run completed for the current uploaded file. Review window outcomes and coverage below.")
    overview_tab, review_tab, evidence_tab = st.tabs(
        ["Overview", "Window review", "Evidence & fingerprint"],
        key="voxtrace_result_tabs",
        on_change="rerun",
    )

    with overview_tab:
        st.header("Processing coverage")
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
        metric_cols = st.columns(4, border=True)
        metric_cols[0].metric("Recording", f"{coverage['recording_duration_seconds']:.2f} s")
        metric_cols[1].metric("Classified time (union)", f"{coverage['classified_all_seconds']:.2f} s")
        metric_cols[2].metric("Classified coverage", f"{coverage['analyzed_time_fraction']:.1%}")
        metric_cols[3].metric("Pinned model", "Verified" if model_ready else "Not verified")
        st.caption(
            f"{coverage['unanalyzed_seconds']:.2f} s had no successful classification; "
            f"{coverage['unprocessed_seconds']:.2f} s had no completed window outcome. "
            f"Model revision: {MODEL_REVISION}. These are processing measures, not prediction-validity measures."
        )

        count_cols = st.columns(5, border=True)
        count_cols[0].metric("Total windows", coverage["total_windows"])
        count_cols[1].metric("Classified", coverage["successfully_classified_windows"])
        count_cols[2].metric("Flagged classified", coverage["flagged_classified_windows"])
        count_cols[3].metric("Skipped", coverage["skipped_windows"])
        count_cols[4].metric("Failed", coverage["failed_windows"])
        time_cols = st.columns(3, border=True)
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

        st.header("Raw class summaries")
        st.info(LOW_SCORE_NOTICE)
        st.caption(SCORE_DESCRIPTION)
        if stats is None:
            st.write(
                "Per-class file summaries are unavailable for this coverage state. "
                "Classified individual windows remain available in Window review."
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
                                "Mean raw score": values["mean"],
                                "Median raw score": values["median"],
                                "Maximum raw score": values["maximum"],
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

    with review_tab:
        st.header("Listen-first candidate review")
        st.subheader(QUEUE_TITLE)
        available_labels = [EXPECTED_ID2LABEL[index] for index in sorted(EXPECTED_ID2LABEL)]
        selectors = list(available_labels)
        if {"angry", "fearful"}.issubset(set(available_labels)):
            selectors.insert(0, "max(angry, fearful)")
        default_selector = "max(angry, fearful)" if "max(angry, fearful)" in selectors else selectors[0]
        selected_selector = st.selectbox(
            "Raw class used to rank the listening queues",
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
                f"{window_heading(candidate)} · queue ranking {candidate['selected_class']} raw score {format_raw_score(candidate['selected_score'])}"
                ):
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
                        f"{window_heading(candidate)} · queue ranking {candidate['selected_class']} raw score {format_raw_score(candidate['selected_score'])}"
                    )
                    show_window(candidate, audio_data)

        st.header("Timestamped window review")
        st.caption(
            "Filter or sort the complete analysis, then choose a row below to update its score, flags, and local playback. "
            "Unflagged means a classified window with no recorded quality flags."
        )
        filter_col, order_col = st.columns([1, 1])
        with filter_col:
            filter_label = st.segmented_control(
                "Window filter",
                ["All windows", "Flagged classified", "Unflagged classified", "Skipped / failed"],
                default="All windows",
                key="voxtrace_window_filter",
            )
        with order_col:
            order_label = st.selectbox(
                "Window order",
                ["Timestamp · earliest first", "Selected raw score · highest first"],
                key="voxtrace_window_order",
            )

        filter_modes = {
            "All windows": "all",
            "Flagged classified": "flagged_classified",
            "Unflagged classified": "unflagged",
            "Skipped / failed": "skipped_failed",
        }
        order_modes = {
            "Timestamp · earliest first": "timestamp",
            "Selected raw score · highest first": "raw_score",
        }
        visible_windows = review_windows(
            windows,
            filter_mode=filter_modes[filter_label],
            order_by=order_modes[order_label],
            selected_label=selected_selector,
            verified_labels=available_labels,
        )
        if not visible_windows:
            st.info("No windows match this filter.")
        else:
            review_rows = []
            for window in visible_windows:
                if window["status"] == "classified":
                    top_output, _, _ = summarize_top_outputs(window["scores"])
                    top_output_text = (
                        f"{top_output['label']} · {format_raw_score(top_output['score'])}"
                    )
                else:
                    top_output_text = "Unavailable"
                review_rows.append({
                    "Window": window["index"] + 1,
                    "Interval": (
                        f"{window['start_seconds']:.2f}–{window['end_seconds']:.2f} s"
                    ),
                    "Status": window["status"],
                    "Top raw output": top_output_text,
                })
            windows_by_index = {int(window["index"]): window for window in visible_windows}
            window_indices = list(windows_by_index)
            selected_key = "voxtrace_window_selected"
            if st.session_state.get(selected_key) not in window_indices:
                st.session_state[selected_key] = window_indices[0]

            def review_choice_label(index: int) -> str:
                window = windows_by_index[index]
                label = window_heading(window)
                if window["status"] == "classified":
                    top_output, _, _ = summarize_top_outputs(window["scores"])
                    label += (
                        f" · {top_output['label']} raw score "
                        f"{format_raw_score(top_output['score'])}"
                    )
                return label

            navigation_col, detail_col = st.columns([2, 3], gap="large")
            with navigation_col:
                st.dataframe(
                    pd.DataFrame(review_rows),
                    hide_index=True,
                    width="stretch",
                    alt="Window review list with interval, processing status, and top raw model output",
                )
                selected_index = st.selectbox(
                    "Select a window to inspect",
                    window_indices,
                    format_func=review_choice_label,
                    key=selected_key,
                )
            selected_window = select_review_window(visible_windows, selected_index)
            with detail_col:
                with st.container(border=True):
                    show_window(selected_window, audio_data, focal=True)

    with evidence_tab:
        st.header("Reproducibility receipt")
        receipt = st.session_state["voxtrace_receipt"]
        receipt_json = strict_json_dumps(receipt)
        with st.container(border=True):
            st.text(f"Original filename: {receipt['input'].get('filename', 'unavailable')}")
            st.caption("Review this filename before sharing the receipt; it can contain identifying metadata.")
            st.write("**Input SHA-256**")
            st.code(receipt["input"]["sha256"], language=None)
            st.caption(
                "The receipt contains the source-byte fingerprint and analysis metadata, but no uploaded audio bytes or arrays. "
                "A matching fingerprint establishes byte equality against a supplied reference only."
            )
            st.download_button(
                "Download JSON receipt",
                data=receipt_json,
                file_name=f"voxtrace_analysis_receipt_{current_hash[:12]}.json",
                mime="application/json",
                key=f"voxtrace_receipt_{receipt_widget_suffix}",
                type="primary",
            )

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
