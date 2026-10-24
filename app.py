
import json
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
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


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="VoxTrace | Audio Triage",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 VoxTrace")
st.caption(
    "AI-assisted speech emotion analysis for investigative triage. "
    "Predictions are model outputs for human review, never proof "
    "of deception, intent, honesty, or guilt."
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading the AI model for the first time...")
def load_classifier():
    return pipeline(
        "audio-classification",
        model=MODEL_NAME,
        device=-1,
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("About VoxTrace")

    st.write("**Analysis:** Overlapping audio windows")
    st.write("**Window:** 3 seconds")
    st.write("**Stride:** 1.5 seconds")
    st.write("**Sample rate:** 16 kHz")
    st.write("**Model:** Wav2Vec2 emotion classifier")

    st.info(
        "This prototype analyzes uploaded audio. "
        "It does not perform live microphone monitoring."
    )

    st.warning(
        "Emotion labels may be wrong. Accent, language, context, "
        "recording quality, music, and non-speech sounds can affect "
        "predictions."
    )

    st.caption(
        "A human reviewer must listen to the original recording "
        "and assess the context."
    )


# --------------------------------------------------
# Upload recording
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an audio recording",
    type=["wav"],
    help="This prototype currently accepts WAV audio files.",
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
            "SHA-256 can reveal changes relative to a previously "
            "recorded fingerprint. It does not establish the source, "
            "identity of the speaker, or authenticity of a recording."
        )

    if st.button("Analyze recording", type="primary"):
        try:
            audio, sample_rate = load_audio(file_bytes)

            with st.spinner("Analyzing audio segments..."):
                classifier = load_classifier()
                results = analyze_audio(
                    audio,
                    classifier,
                    sample_rate,
                )

            # Save results so they remain available after Streamlit reruns.
            st.session_state["voxtrace_results"] = results
            st.session_state["voxtrace_hash"] = file_hash
            st.session_state["voxtrace_bytes"] = file_bytes
            st.session_state["voxtrace_filename"] = uploaded_file.name
            st.session_state["voxtrace_audio"] = audio
            st.session_state["voxtrace_sample_rate"] = sample_rate

        except Exception as exc:
            st.error(f"Could not analyze this file: {exc}")


# --------------------------------------------------
# Analysis results
# --------------------------------------------------

if "voxtrace_results" in st.session_state:
    results = st.session_state["voxtrace_results"]
    audio = st.session_state["voxtrace_audio"]
    sample_rate = st.session_state["voxtrace_sample_rate"]

    st.divider()
    st.header("Analysis results")

    analyzed = [
        item for item in results
        if item.get("label") is not None
    ]

    skipped = len(results) - len(analyzed)

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Total windows", len(results))
    col2.metric("Windows with predictions", len(analyzed))
    col3.metric("Windows without predictions", skipped)

    st.info(
        "A returned prediction does not confirm that a window contains "
        "speech. Model outputs must be reviewed against the audio."
    )

    # --------------------------------------------------
    # Prediction timeline
    # --------------------------------------------------

    if analyzed:
        st.subheader("Emotion prediction timeline")

        df = pd.DataFrame(analyzed)
        df["confidence_percent"] = df["confidence"] * 100

        hover_columns = {
            "end_seconds": True,
            "confidence_percent": ":.1f",
            "rms_energy": True,
            "peak_amplitude": True,
        }

        if "spectral_flatness" in df.columns:
            hover_columns["spectral_flatness"] = True

        chart = px.scatter(
            df,
            x="start_seconds",
            y="label",
            size="confidence_percent",
            color="label",
            hover_data=hover_columns,
            labels={
                "start_seconds": "Window start (seconds)",
                "end_seconds": "Window end (seconds)",
                "label": "Predicted label",
                "confidence_percent": "Model score (%)",
                "rms_energy": "RMS energy",
                "peak_amplitude": "Peak amplitude",
                "spectral_flatness": "Spectral flatness",
            },
            title="Timestamped model predictions",
        )

        chart.update_layout(
            xaxis_title="Time in recording (seconds)",
            yaxis_title="Predicted emotion label",
            legend_title_text="Model label",
        )

        st.plotly_chart(chart, width="stretch")

        st.caption(
            "Model scores are outputs of the classifier. They are not "
            "proof of an emotional state and are not probabilities "
            "of truthfulness, deception, or guilt."
        )

    else:
        st.warning(
            "No windows received predictions. Try a clearer recording "
            "with audible speech."
        )

    # --------------------------------------------------
    # Audio bookmarks and human review
    # --------------------------------------------------

    st.subheader("Acoustic-anchored review bookmarks")

    st.write(
        "Expand a window to listen to its audio, review the model scores, "
        "and inspect acoustic indicators. These indicators cannot "
        "reliably distinguish speech from tones, music, or noise."
    )

    for index, item in enumerate(results):
        start = item["start_seconds"]
        end = item["end_seconds"]

        label = item.get("label") or item.get(
            "status", "No prediction"
        )

        heading = f"{start:.2f}s–{end:.2f}s | {label}"

        with st.expander(heading):
            st.write(f"**Window:** {index + 1}")
            st.write(f"**Status:** {item.get('status', 'Not specified')}")

            if item.get("label") is not None:
                st.write(f"**Leading model label:** {item['label']}")

            if item.get("confidence") is not None:
                st.write(
                    f"**Leading model score:** "
                    f"{item['confidence'] * 100:.1f}%"
                )

            st.write(
                f"**RMS energy:** {item.get('rms_energy', 'N/A')}"
            )
            st.write(
                f"**Peak amplitude:** "
                f"{item.get('peak_amplitude', 'N/A')}"
            )

            if item.get("spectral_flatness") is not None:
                st.write(
                    f"**Spectral flatness:** "
                    f"{item['spectral_flatness']}"
                )

            # Acoustic heuristic warnings
            review_flags = item.get("review_flags", [])

            if review_flags:
                st.warning(
                    "Review flags — these are heuristic indicators, "
                    "not proof that speech is absent:\n\n"
                    + "\n".join(
                        f"- {flag}" for flag in review_flags
                    )
                )
            else:
                st.caption(
                    "No acoustic heuristic flags were generated. "
                    "This does not confirm that the window contains speech."
                )

            # Play the corresponding audio segment
            start_sample = int(start * sample_rate)
            end_sample = min(
                int(end * sample_rate),
                len(audio),
            )

            snippet = audio[start_sample:end_sample]

            if len(snippet) > 0:
                st.audio(
                    snippet,
                    sample_rate=sample_rate,
                )

            # Top three predictions
            top_predictions = item.get("top_predictions", [])

            if top_predictions:
                st.markdown("**Top 3 model predictions**")

                top_df = pd.DataFrame(top_predictions)
                top_df["score_percent"] = top_df["score"] * 100

                st.dataframe(
                    top_df[["label", "score_percent"]].rename(
                        columns={
                            "label": "Predicted label",
                            "score_percent": "Model score (%)",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )

            # Full score distribution
            scores = item.get("scores", [])

            if scores:
                with st.expander("View all model scores"):
                    score_df = pd.DataFrame(scores)
                    score_df["score_percent"] = (
                        score_df["score"] * 100
                    )

                    st.dataframe(
                        score_df[["label", "score_percent"]].rename(
                            columns={
                                "label": "Model label",
                                "score_percent": "Model score (%)",
                            }
                        ),
                        hide_index=True,
                        width="stretch",
                    )

    # --------------------------------------------------
    # Download JSON analysis receipt
    # --------------------------------------------------

    st.subheader("Download analysis receipt")

    receipt = build_receipt(
        st.session_state["voxtrace_bytes"],
        st.session_state["voxtrace_filename"],
        results,
    )

    receipt["generated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    st.download_button(
        "Download JSON receipt",
        data=json.dumps(receipt, indent=2),
        file_name="voxtrace_analysis_receipt.json",
        mime="application/json",
    )

    # --------------------------------------------------
    # SHA-256 fingerprint verification
    # --------------------------------------------------

    st.subheader("Verify a previously recorded fingerprint")

    expected_hash = st.text_input(
        "Paste the expected SHA-256 fingerprint",
        value=st.session_state["voxtrace_hash"],
        help=(
            "Enter the fingerprint recorded earlier for the original "
            "file. This verifies byte equality against that fingerprint, "
            "not recording authenticity."
        ),
    )

    if st.button("Verify uploaded recording"):
        candidate_hash = expected_hash.strip()

        if (
            len(candidate_hash) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in candidate_hash
            )
        ):
            st.error(
                "Enter a valid SHA-256 fingerprint containing "
                "exactly 64 hexadecimal characters."
            )
        else:
            matches = verify_sha256(
                st.session_state["voxtrace_bytes"],
                candidate_hash.lower(),
            )

            if matches:
                st.success(
                    "The uploaded file bytes match the supplied fingerprint."
                )
            else:
                st.error(
                    "The uploaded file bytes do not match the supplied fingerprint."
                )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "VoxTrace is a prototype for human-assisted investigative triage. "
    "It is not a lie detector, diagnostic tool, speaker-identification "
    "system, or substitute for professional investigation."
)
