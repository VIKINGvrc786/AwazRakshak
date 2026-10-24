
import hashlib
import io

import librosa
import numpy as np
import soundfile as sf
from transformers import pipeline


MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 3.0
STRIDE_SECONDS = 1.5
MIN_AUDIO_SECONDS = 0.5
SILENCE_RMS_THRESHOLD = 0.005


def calculate_sha256(file_bytes: bytes) -> str:
    """Fingerprint the exact uploaded file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def verify_sha256(file_bytes: bytes, expected_hash: str) -> bool:
    """Check whether bytes match a previously recorded fingerprint."""
    return calculate_sha256(file_bytes) == expected_hash


def load_audio(file_bytes: bytes):
    """Load audio, convert to mono, and resample to 16 kHz."""
    if not file_bytes:
        raise ValueError("The uploaded audio file is empty.")

    try:
        audio, sample_rate = sf.read(
            io.BytesIO(file_bytes),
            dtype="float32",
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise ValueError(
            "Unable to read this audio file. Please upload a supported format."
        ) from exc

    if sample_rate <= 0:
        raise ValueError("The audio has an invalid sample rate.")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError("The uploaded audio contains no usable samples.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("The audio contains invalid numeric values.")

    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        )

    if len(audio) < int(MIN_AUDIO_SECONDS * SAMPLE_RATE):
        raise ValueError("Please upload at least 0.5 seconds of audio.")

    return audio.astype(np.float32), SAMPLE_RATE


def inspect_audio_segment(segment, sample_rate):
    """
    Calculate simple acoustic indicators.

    These indicators can flag possible tonal audio, but cannot reliably
    determine whether a segment contains speech.
    """
    rms = float(np.sqrt(np.mean(np.square(segment))))
    peak = float(np.max(np.abs(segment)))

    flags = []

    # Inspect spectral flatness over short frames.
    # Very low flatness can indicate tonal audio, but voiced speech
    # and music may also have low flatness.
    frame_length = 1024
    hop_length = 512

    if len(segment) >= frame_length:
        flatness = librosa.feature.spectral_flatness(
            y=segment,
            n_fft=frame_length,
            hop_length=hop_length,
        )
        mean_flatness = float(np.mean(flatness))
    else:
        mean_flatness = None

    if mean_flatness is not None and mean_flatness < 0.01:
        flags.append("Possible tonal audio — verify that speech is present")

    return {
        "rms_energy": round(rms, 5),
        "peak_amplitude": round(peak, 5),
        "spectral_flatness": (
            round(mean_flatness, 5)
            if mean_flatness is not None
            else None
        ),
        "review_flags": flags,
    }


def analyze_audio(audio, classifier, sample_rate=SAMPLE_RATE):
    """Analyze overlapping audio windows and retain timestamped results."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if audio is None or len(audio) == 0:
        raise ValueError("No audio samples were provided.")

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
        raise ValueError("Audio must be a finite, mono waveform.")

    window_size = int(WINDOW_SECONDS * sample_rate)
    stride_size = int(STRIDE_SECONDS * sample_rate)

    results = []
    start = 0

    while start < len(audio):
        end = min(start + window_size, len(audio))
        segment = audio[start:end]

        # Ignore very short trailing fragments.
        if len(segment) < int(MIN_AUDIO_SECONDS * sample_rate):
            break

        inspection = inspect_audio_segment(segment, sample_rate)
        rms = inspection["rms_energy"]

        result = {
            "start_seconds": round(start / sample_rate, 2),
            "end_seconds": round(end / sample_rate, 2),
            "duration_seconds": round(len(segment) / sample_rate, 2),
            **inspection,
            "label": None,
            "confidence": None,
            "top_predictions": [],
            "scores": [],
            "status": None,
        }

        if rms < SILENCE_RMS_THRESHOLD:
            result["status"] = "Skipped: low-energy audio"
            result["review_flags"].append(
                "Possible silence or very quiet audio"
            )
            results.append(result)
            start += stride_size
            continue

        predictions = classifier(
            {"raw": segment, "sampling_rate": sample_rate},
            top_k=7,
        )

        # Keep the complete score list for compatibility and transparency.
        scores = [
            {
                "label": item["label"],
                "score": float(item["score"]),
            }
            for item in predictions
        ]
        scores.sort(key=lambda item: item["score"], reverse=True)

        result["scores"] = scores
        result["top_predictions"] = scores[:3]
        result["label"] = scores[0]["label"] if scores else None
        result["confidence"] = scores[0]["score"] if scores else None

        result["status"] = (
            "Model prediction — human review required"
        )

        if not scores:
            result["status"] = "No predictions returned"
            result["review_flags"].append(
                "The model returned no prediction scores"
            )

        results.append(result)
        start += stride_size

    return results


def build_receipt(file_bytes: bytes, filename: str, results: list):
    """Create a transparent, downloadable analysis record."""
    return {
        "application": "VoxTrace",
        "filename": filename,
        "sha256": calculate_sha256(file_bytes),
        "model": MODEL_NAME,
        "sample_rate_hz": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "analysis_method": (
            "Overlapping audio windows with model predictions "
            "and basic acoustic indicators"
        ),
        "results": results,
        "limitations": [
            "Emotion predictions are indicative model outputs, not facts.",
            "Model scores are not calibrated proof of emotional state.",
            "Emotion is not proof of deception, intent, honesty, or guilt.",
            "Acoustic indicators cannot reliably distinguish speech "
            "from music, tones, or other non-speech audio.",
            "Low-energy segments are skipped using a configurable threshold.",
            "SHA-256 detects changes relative to a saved hash; it does "
            "not prove recording authenticity or identify the speaker.",
            "A human reviewer must assess the original audio and context.",
        ],
    }


if __name__ == "__main__":
    original = b"VoxTrace test recording"
    original_hash = calculate_sha256(original)

    print("VoxTrace — Integrity Check")
    print("Original verified:", verify_sha256(original, original_hash))
    print(
        "Modified verified:",
        verify_sha256(b"Altered recording", original_hash),
    )
