
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import pipeline

from engine import (
    MODEL_NAME,
    SAMPLE_RATE,
    load_audio,
    analyze_audio,
    build_receipt,
    calculate_sha256,
    verify_sha256,
)


st.set_page_config(
    page_title="VoxTrace | Audio Triage",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 VoxTrace")
st.caption(
    "AI-assisted speech emotion analysis for investigative triage. "
    "Predictions are indicators for human review, never proof of deception or guilt."
)


@st.cache_resource(show_spinner="Loading the AI model for the first time...")
def load_classifier():
    return pipeline(
        "audio-classification",
        model=MODEL_NAME,
        device=-1,
    )


with st.sidebar:
    st.header("About VoxTrace")
    st.write("**Analysis:** Overlapping audio windows")
    st.write("**Window:** 3 seconds")
    st.write("**Stride:** 1.5 seconds")
    st.write("**Model:** Wav2Vec2 emotion classifier")
    st.info(
        "The current version analyzes uploaded audio. "
        "It is not live microphone monitoring."
    )
    st.warning(
        "Emotion labels can be wrong. Accent, language, recording quality, "
        "and context may affect predictions."
    )


uploaded_file = st.file_uploader(
    "Upload an audio recording",
    type=["wav"],
    help="For this first version, upload a WAV file.",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = calculate_sha256(file_bytes)

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Recording")
        st.audio(file_bytes, format="audio/wav")
        st.write(f"**Filename:** {uploaded_file.name}")

    with right:
        st.subheader("Integrity fingerprint")
        st.code(file_hash, language=None)
        st.caption(
            "This fingerprint can reveal later changes relative to a saved hash. "
            "It does not establish the recording's origin or authenticity."
        )

    if st.button("Analyze recording", type="primary"):
        try:
            audio, sample_rate = load_audio(file_bytes)

            with st.spinner("Analyzing audio segments..."):
                classifier = load_classifier()
                results = analyze_audio(audio, classifier, sample_rate)

            st.session_state["voxtrace_results"] = results
            st.session_state["voxtrace_hash"] = file_hash
            st.session_state["voxtrace_bytes"] = file_bytes
            st.session_state["voxtrace_filename"] = uploaded_file.name
            st.session_state["voxtrace_audio"] = audio
            st.session_state["voxtrace_sample_rate"] = sample_rate

        except Exception as exc:
            st.error(f"Could not analyze this file: {exc}")


if "voxtrace_results" in st.session_state:
    results = st.session_state["voxtrace_results"]
    audio = st.session_state["voxtrace_audio"]
    sample_rate = st.session_state["voxtrace_sample_rate"]

    st.divider()
    st.header("Analysis results")

    analyzed = [r for r in results if r["label"] is not None]
    skipped = len(results) - len(analyzed)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total windows", len(results))
    c2.metric("Windows with predictions", len(analyzed))
    c3.metric("Low-energy windows skipped", skipped)

    if analyzed:
        df = pd.DataFrame(analyzed)
        df["confidence_percent"] = df["confidence"] * 100

        st.subheader("Emotion prediction timeline")
        chart = px.scatter(
            df,
            x="start_seconds",
            y="label",
            size="confidence_percent",
            color="label",
            hover_data={
                "end_seconds": True,
                "confidence_percent": ":.1f",
                "rms_energy": True,
                "peak_amplitude": True,
            },
            labels={
                "start_seconds": "Window start (seconds)",
                "label": "Predicted label",
                "confidence_percent": "Model score (%)",
            },
            title="Timestamped model predictions",
        )
        st.plotly_chart(chart, use_container_width=True)

        st.caption(
            "The model score is not a probability that a person is telling the truth "
            "or experiencing an emotion. Treat it as a model output for review."
        )

        st.subheader("Acoustic-anchored review bookmarks")
        st.write(
            "Open a segment to listen to the corresponding audio. "
            "The energy values are simple acoustic measurements, not emotion evidence."
        )

        for index, item in enumerate(results):
            start = item["start_seconds"]
            end = item["end_seconds"]

            label = item["label"] or item["status"]
            heading = f"{start:.2f}s–{end:.2f}s | {label}"

            with st.expander(heading):
                st.write(f"**Status:** {item['status']}")
                st.write(f"**RMS energy:** {item['rms_energy']}")
                st.write(f"**Peak amplitude:** {item['peak_amplitude']}")

                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                snippet = audio[start_sample:end_sample]

                if len(snippet):
                    st.audio(snippet, sample_rate=sample_rate)

                if item["scores"]:
                    score_df = pd.DataFrame(item["scores"])
                    score_df["score_percent"] = score_df["score"] * 100
                    st.dataframe(
                        score_df[["label", "score_percent"]].rename(
                            columns={
                                "label": "Model label",
                                "score_percent": "Model score (%)",
                            }
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

    else:
        st.warning(
            "No windows received predictions. Try a clearer recording with audible speech."
        )

    st.subheader("Download analysis receipt")

    receipt = build_receipt(
        st.session_state["voxtrace_bytes"],
        st.session_state["voxtrace_filename"],
        results,
    )
    receipt["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    st.download_button(
        "Download JSON receipt",
        data=json.dumps(receipt, indent=2),
        file_name="voxtrace_analysis_receipt.json",
        mime="application/json",
    )

    st.subheader("Verify a previously recorded fingerprint")
    expected_hash = st.text_input(
        "Paste the expected SHA-256 fingerprint",
        value=st.session_state["voxtrace_hash"],
    )

    if st.button("Verify uploaded recording"):
        if len(expected_hash.strip()) != 64:
            st.error("A SHA-256 fingerprint should contain 64 hexadecimal characters.")
        else:
            matches = verify_sha256(
                st.session_state["voxtrace_bytes"],
                expected_hash.strip().lower(),
            )
            if matches:
                st.success("The uploaded bytes match the supplied fingerprint.")
            else:
                st.error("The uploaded bytes do not match the supplied fingerprint.")

st.divider()
st.caption(
    "VoxTrace is a prototype for human-assisted triage, not a lie detector, "
    "diagnostic tool, or substitute for professional investigation."
)
