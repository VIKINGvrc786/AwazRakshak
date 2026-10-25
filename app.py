"""AwazRakshak's single-page live microphone workflow."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from config import (
    INDICATOR_FORMULAS,
    MODEL_ID,
    MODEL_LOCAL_PATH_ENV,
    MODEL_REVISION,
)
from engine import ModelBundle, load_model, pinned_model_cache_status
from live import LiveCallPipeline

try:
    from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer
except ImportError as exc:
    AudioProcessorBase = None
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_IMPORT_ERROR = str(exc)
else:
    WEBRTC_IMPORT_ERROR = None


st.set_page_config(page_title="AwazRakshak", page_icon="◌", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background: #F4F8F8;
        color: #24383B;
    }
    .stApp p {
        color: #24383B !important;
    }
    [data-testid="stMetric"] {
        background: #E3EEEE;
        border: 1px solid #C9DCDC;
        border-radius: 14px;
        padding: 1rem;
    }
    [data-testid="stCaptionContainer"] p {
        color: #24383B !important;
    }
    .stButton > button, .stDownloadButton > button {
        background: #35696E;
        color: #FFFFFF;
        border: 1px solid #35696E;
        border-radius: 12px;
        min-height: 2.8rem;
    }
    .stButton > button p, .stDownloadButton > button p {
        color: #FFFFFF !important;
    }
    .attention-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        color: #24383B;
        font-weight: 600;
    }
    .attention-routine { background: #8FB8A8; }
    .attention-elevated { background: #E8C37A; }
    .attention-high { background: #E39A84; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=15, max_entries=8, show_spinner=False)
def check_model(path: str | None) -> tuple[bool, str]:
    return pinned_model_cache_status(path or None)


@st.cache_resource(max_entries=2, show_spinner=False)
def load_verified_model(path: str | None) -> ModelBundle:
    return load_model(path or None)


def make_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    insights = snapshot["insights"]
    flags = sorted(
        {
            flag
            for insight in insights
            for flag in insight["quality_flags"]
        }
    )
    return {
        "application": "AwazRakshak",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": snapshot["started_at"],
        "duration_seconds": round(snapshot["duration_seconds"], 1),
        "analyzed_windows": len(insights),
        "final_attention_category": (
            insights[-1]["category"] if insights else "No analysis available"
        ),
        "peak_distress_indicator": round(
            max(
                (
                    insight["indicators"]["distress_indicator"]
                    for insight in insights
                ),
                default=0.0,
            ),
            4,
        ),
        "peak_nervousness_indicator": round(
            max(
                (
                    insight["indicators"]["nervousness_indicator"]
                    for insight in insights
                ),
                default=0.0,
            ),
            4,
        ),
        "quality_flags": flags,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
    }


def save_summary(summary: dict[str, Any]) -> Path:
    directory = Path(__file__).resolve().parent / ".awazrakshak-data" / "summaries"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = directory / f"call-summary-{timestamp}.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return output_path


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "recording" not in st.session_state:
    st.session_state.recording = False
if "summary" not in st.session_state:
    st.session_state.summary = None
if "summary_path" not in st.session_state:
    st.session_state.summary_path = None

st.title("AwazRakshak")
st.caption("Local, real-time voice emotion signals")
with st.container(border=True):
    st.markdown(
        "Indicative signals to support human judgement; not a decision about danger, "
        "deception, guilt or dispatch."
    )

local_path = st.text_input(
    "Local model path",
    value=os.environ.get(MODEL_LOCAL_PATH_ENV, ""),
    placeholder="Hugging Face cache root or pinned snapshot folder",
    help=(
        "Select the Hugging Face cache root, pinned snapshot folder, or a model "
        "folder containing config.json, preprocessor_config.json, and model.safetensors."
    ),
)
selected_path = local_path.strip() or None
authorized = st.checkbox("I am authorised to record this call.")

model_available, model_error = check_model(selected_path)
if model_available:
    st.caption("Pinned local model verified. Live analysis is available.")
else:
    st.warning(f"Analysis disabled: {model_error}")

if WEBRTC_IMPORT_ERROR:
    st.warning(
        "Browser microphone capture is unavailable because streamlit-webrtc "
        f"could not be loaded: {WEBRTC_IMPORT_ERROR}"
    )

start_column, stop_column = st.columns(2, vertical_alignment="bottom")
with start_column:
    start_clicked = st.button(
        "Start recording",
        disabled=(
            not authorized
            or not model_available
            or WEBRTC_IMPORT_ERROR is not None
            or st.session_state.recording
        ),
        width="stretch",
    )
with stop_column:
    stop_clicked = st.button(
        "Stop and save summary",
        disabled=not st.session_state.recording,
        width="stretch",
    )

if start_clicked:
    st.session_state.pipeline = LiveCallPipeline(
        lambda: load_verified_model(selected_path)
    )
    st.session_state.pipeline.start()
    st.session_state.recording = True
    st.session_state.summary = None
    st.session_state.summary_path = None

if stop_clicked:
    pipeline = st.session_state.pipeline
    if pipeline is not None:
        pipeline.stop()
        st.session_state.summary = make_summary(pipeline.snapshot())
        try:
            st.session_state.summary_path = str(
                save_summary(st.session_state.summary)
            )
        except OSError as exc:
            st.session_state.summary_path = None
            st.warning(f"Could not save the call summary: {exc}")
    st.session_state.recording = False
    st.session_state.pipeline = None


if st.session_state.recording:
    pipeline: LiveCallPipeline = st.session_state.pipeline
    if AudioProcessorBase is not None:

        class MicrophoneProcessor(AudioProcessorBase):
            def recv(self, frame: Any) -> Any:
                pipeline.add_audio(frame.to_ndarray(), frame.sample_rate)
                return frame

        webrtc_streamer(
            key="awazrakshak-microphone",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=MicrophoneProcessor,
            media_stream_constraints={"audio": True, "video": False},
            desired_playing_state=True,
        )
        st.caption("Allow microphone access in your browser to begin analysis.")

    @st.fragment(run_every="1.5s")
    def live_panel() -> None:
        snapshot = st.session_state.pipeline.snapshot()
        latest = snapshot["latest"]
        error = snapshot["error"]
        if error:
            st.warning(error)
        elif latest is None:
            st.caption("Listening for a 3-second audio window...")
        else:
            st.subheader("Live insight")
            best = max(latest["scores"], key=lambda item: item["score"])
            st.write(
                f"Top raw model label (uncalibrated): **{best['label']}** "
                f"({best['score']:.2f})"
            )
            st.bar_chart(
                {
                    item["label"]: item["score"]
                    for item in latest["scores"]
                },
                alt="Uncalibrated raw model score for each of the seven labels.",
            )

            distress_column, nervousness_column = st.columns(2)
            distress_column.metric(
                "Distress indicator",
                f"{latest['indicators']['distress_indicator']:.2f}",
                help="Derived indicator (not a model class).",
            )
            nervousness_column.metric(
                "Nervousness indicator",
                f"{latest['indicators']['nervousness_indicator']:.2f}",
                help="Derived indicator (not a model class).",
            )
            st.caption(
                "Derived indicator (not a model class). "
                f"Formulas: {INDICATOR_FORMULAS['distress_indicator']}; "
                f"{INDICATOR_FORMULAS['nervousness_indicator']}."
            )

            if latest["low_confidence"]:
                st.warning("Low confidence: audio quality")
            else:
                st.markdown(
                    "<span class='attention-badge attention-"
                    f"{latest['category'].lower()}'>Indicative attention: "
                    f"{latest['category']}</span>",
                    unsafe_allow_html=True,
                )
            if latest["quality_flags"]:
                st.caption("Audio quality flags: " + ", ".join(latest["quality_flags"]))

            trend = [
                {
                    "Distress indicator": insight["indicators"]["distress_indicator"],
                    "Nervousness indicator": insight["indicators"][
                        "nervousness_indicator"
                    ],
                }
                for insight in snapshot["insights"]
            ]
            st.line_chart(
                trend,
                alt="Live trend of the derived distress and nervousness indicators.",
            )
            if snapshot["events"]:
                st.caption("Attention category changes")
                for event in snapshot["events"][-8:]:
                    elapsed = int(event["time"])
                    st.write(
                        f"{elapsed // 60:02d}:{elapsed % 60:02d} "
                        f"{event['from']} → {event['to']}"
                    )

    live_panel()


if st.session_state.summary:
    summary = st.session_state.summary
    st.subheader("Call summary")
    st.write(
        f"Duration: {summary['duration_seconds']:.1f} seconds · "
        f"Analyzed windows: {summary['analyzed_windows']}"
    )
    st.write(f"Final indicative category: {summary['final_attention_category']}")
    st.write(
        f"Peak distress: {summary['peak_distress_indicator']:.2f} · "
        f"Peak nervousness: {summary['peak_nervousness_indicator']:.2f}"
    )
    if summary["quality_flags"]:
        st.caption("Recorded audio quality flags: " + ", ".join(summary["quality_flags"]))
    if st.session_state.summary_path:
        st.caption(f"Saved summary: {st.session_state.summary_path}")
    st.download_button(
        "Download summary JSON",
        data=json.dumps(summary, indent=2),
        file_name="awazrakshak-call-summary.json",
        mime="application/json",
        width="stretch",
    )
