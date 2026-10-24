
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


def calculate_sha256(file_bytes: bytes) -> str:
    """Fingerprint the exact uploaded file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def verify_sha256(file_bytes: bytes, expected_hash: str) -> bool:
    """Check whether bytes match a previously recorded fingerprint."""
    return calculate_sha256(file_bytes) == expected_hash


def load_audio(file_bytes: bytes):
    """Load audio, convert to mono, and resample to 16 kHz."""
    audio, sample_rate = sf.read(io.BytesIO(file_bytes), dtype="float32")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if len(audio) == 0:
        raise ValueError("The uploaded audio is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("The audio contains invalid numeric values.")

    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        )

    if len(audio) < int(0.5 * SAMPLE_RATE):
        raise ValueError("Please upload at least 0.5 seconds of audio.")

    return audio, SAMPLE_RATE


def analyze_audio(audio, classifier, sample_rate=SAMPLE_RATE):
    """Analyze overlapping audio windows and retain timestamped results."""
    window_size = int(WINDOW_SECONDS * sample_rate)
    stride_size = int(STRIDE_SECONDS * sample_rate)

    results = []
    start = 0

    while start < len(audio):
        end = min(start + window_size, len(audio))
        segment = audio[start:end]

        # Skip very short trailing fragments.
        if len(segment) < int(0.5 * sample_rate):
            break

        rms = float(np.sqrt(np.mean(np.square(segment))))
        peak = float(np.max(np.abs(segment)))

        result = {
            "start_seconds": round(start / sample_rate, 2),
            "end_seconds": round(end / sample_rate, 2),
            "duration_seconds": round(len(segment) / sample_rate, 2),
            "rms_energy": round(rms, 5),
            "peak_amplitude": round(peak, 5),
            "label": None,
            "confidence": None,
            "status": None,
            "scores": [],
        }

        # This is a simple configurable silence gate, not a
        # universal threshold for every recording environment.
        if rms < 0.005:
            result["status"] = "Skipped: low-energy audio"
            results.append(result)
            start += stride_size
            continue

        predictions = classifier(
            {"raw": segment, "sampling_rate": sample_rate},
            top_k=7,
        )

        result["scores"] = [
            {
                "label": item["label"],
                "score": float(item["score"]),
            }
            for item in predictions
        ]

        result["label"] = predictions[0]["label"]
        result["confidence"] = float(predictions[0]["score"])
        result["status"] = "Model prediction — human review required"

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
        "results": results,
        "limitations": [
            "Emotion predictions are indicative model outputs, not facts.",
            "Emotion is not proof of deception, intent, honesty, or guilt.",
            "Low-energy segments are skipped using a configurable threshold.",
            "SHA-256 detects changes relative to a saved hash; it does not prove recording authenticity.",
            "A human reviewer must assess the audio and its context.",
        ],
    }


if __name__ == "__main__":
    original = b"VoxTrace test recording"
    original_hash = calculate_sha256(original)

    print("VoxTrace — Integrity Check")
    print("Original verified:", verify_sha256(original, original_hash))
    print("Modified verified:", verify_sha256(b"Altered recording", original_hash))
