"""Local WAV processing and pinned Wav2Vec2 inference."""

from __future__ import annotations

import importlib.metadata
import io
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf
import soxr

from config import (
    CLIPPING_PEAK_THRESHOLD,
    DIGITAL_SILENCE_RMS_FLOOR,
    EXPECTED_ID2LABEL,
    EXPECTED_LABEL2ID,
    LOW_RMS_FLAG_THRESHOLD,
    LOW_VAD_ACTIVITY_RATIO,
    MAX_UPLOAD_MB,
    MIN_ANALYZABLE_WINDOW_SECONDS,
    MODEL_ID,
    MODEL_OUTPUT_DIMENSION,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    SCORE_SUM_TOLERANCE,
    STATIONARY_NOISE_RMS_CV_MAX,
    STATIONARY_NOISE_SPECTRAL_FLATNESS_MIN,
    TONAL_DOMINANT_POWER_FRACTION,
    TONAL_SPECTRAL_FLATNESS_MAX,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SECONDS,
)
from signals import class_statistics, coverage_accounting


class AudioValidationError(ValueError):
    """An uploaded WAV cannot be safely decoded or analyzed."""


class ModelVerificationError(RuntimeError):
    """The pinned model failed a runtime integrity or compatibility check."""


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    original_wav_metadata: dict[str, Any]
    transform: dict[str, Any]


@dataclass
class ModelBundle:
    model: Any
    feature_extractor: Any
    revision: str
    labels: tuple[str, ...]
    output_dimension: int
    sample_rate: int
    verification: dict[str, Any]

    def predict_scores(self, waveform: np.ndarray) -> list[dict[str, Any]]:
        import torch

        batch = self.feature_extractor(
            waveform.astype(np.float32, copy=False),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = self.model(**batch).logits
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ModelVerificationError(
                    f"Model returned unexpected logits shape {tuple(logits.shape)}"
                )
            if int(logits.shape[-1]) != self.output_dimension:
                raise ModelVerificationError(
                    "Model output dimension changed after verification"
                )
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
        raw = [
            {"label": label, "score": score}
            for label, score in zip(self.labels, probabilities, strict=True)
        ]
        return validate_score_vector(raw, self.labels, self.output_dimension)


def _normalize_id2label(mapping: Mapping) -> dict[int, str]:
    try:
        return {int(index): str(label) for index, label in mapping.items()}
    except (TypeError, ValueError, AttributeError) as exc:
        raise ModelVerificationError("Model id2label is not a valid mapping") from exc


def validate_label_mapping(
    id2label: Mapping,
    label2id: Mapping,
    num_labels: int,
    *,
    expected_id2label: Mapping[int, str] | None = None,
) -> tuple[str, ...]:
    """Require exact ids, raw names, inverse map, and configured class count."""
    normalized = _normalize_id2label(id2label)
    inverse = {str(label): int(index) for label, index in label2id.items()}
    expected = dict(expected_id2label or EXPECTED_ID2LABEL)
    if normalized != expected:
        raise ModelVerificationError(
            f"Raw id2label mismatch: expected {expected}, observed {normalized}"
        )
    if inverse != {label: index for index, label in expected.items()}:
        raise ModelVerificationError("Raw label2id is not the exact inverse of id2label")
    if int(num_labels) != len(expected) or int(num_labels) != MODEL_OUTPUT_DIMENSION:
        raise ModelVerificationError(
            f"num_labels={num_labels} does not match configured dimension {MODEL_OUTPUT_DIMENSION}"
        )
    return tuple(expected[index] for index in range(len(expected)))


def validate_score_vector(
    raw_scores: Any,
    expected_labels: Sequence[str],
    output_dimension: int,
    *,
    tolerance: float = SCORE_SUM_TOLERANCE,
) -> list[dict[str, Any]]:
    """Validate complete softmax output without defaulting missing classes."""
    from numbers import Real

    if not isinstance(raw_scores, (list, tuple)):
        raise ValueError("Classifier output must be a list of label-score objects")
    if output_dimension != len(expected_labels) or len(raw_scores) != output_dimension:
        raise ValueError("Classifier output class count does not match verified dimension")
    parsed: dict[str, float] = {}
    for item in raw_scores:
        if not isinstance(item, Mapping) or not {"label", "score"}.issubset(item):
            raise ValueError("Each classifier output must contain label and score")
        label = item["label"]
        if not isinstance(label, str) or label not in expected_labels:
            raise ValueError(f"Unexpected raw classifier label: {label!r}")
        if label in parsed:
            raise ValueError(f"Duplicate raw classifier label: {label!r}")
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError(f"Score for {label!r} is not a real number")
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Score for {label!r} must be finite and within [0, 1]")
        parsed[label] = score

    if set(parsed) != set(expected_labels):
        missing = sorted(set(expected_labels) - set(parsed))
        raise ValueError(f"Classifier output is missing expected labels: {missing}")
    total = sum(parsed.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"Softmax scores sum to {total:.8f}, outside tolerance {tolerance}")
    return [{"label": label, "score": parsed[label]} for label in expected_labels]


def load_audio(file_bytes: bytes) -> AudioData:
    """Decode a WAV from its original bytes, validate it, mono-mix and resample."""
    if not isinstance(file_bytes, (bytes, bytearray)) or not file_bytes:
        raise AudioValidationError("The uploaded WAV is empty")
    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise AudioValidationError(f"WAV exceeds the {MAX_UPLOAD_MB} MB upload limit")

    try:
        with io.BytesIO(file_bytes) as stream:
            info = sf.info(stream)
            stream.seek(0)
            decoded, source_rate = sf.read(
                stream,
                dtype="float32",
                always_2d=True,
            )
    except Exception as exc:
        raise AudioValidationError(
            "The upload is not a supported, readable WAV file"
        ) from exc

    decoded = np.asarray(decoded, dtype=np.float32)
    if decoded.ndim != 2 or decoded.shape[0] == 0 or decoded.shape[1] == 0:
        raise AudioValidationError("The WAV contains no usable audio samples")
    if not np.isfinite(decoded).all():
        raise AudioValidationError(
            "The WAV contains non-finite samples; the recording was rejected before resampling"
        )
    if source_rate <= 0 or info.channels <= 0:
        raise AudioValidationError("The WAV has invalid sampling-rate or channel metadata")

    original_duration = float(info.frames / source_rate)
    if original_duration < MIN_ANALYZABLE_WINDOW_SECONDS:
        raise AudioValidationError(
            f"WAV is too short to analyze safely (minimum {MIN_ANALYZABLE_WINDOW_SECONDS:.1f} seconds)"
        )

    mono = decoded.mean(axis=1, dtype=np.float32) if info.channels > 1 else decoded[:, 0]
    reduced_channels = info.channels > 1
    if source_rate == MODEL_SAMPLE_RATE:
        converted = mono.astype(np.float32, copy=False)
        resampled = False
    else:
        converted = soxr.resample(
            mono,
            source_rate,
            MODEL_SAMPLE_RATE,
            quality="HQ",
        ).astype(np.float32, copy=False)
        resampled = True
    if converted.size == 0 or not np.isfinite(converted).all():
        raise AudioValidationError("Audio conversion produced no finite usable samples")

    metadata = {
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate_hz": int(source_rate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": original_duration,
    }
    transform = {
        "channel_reduction": "arithmetic mean to mono" if reduced_channels else "none; input already mono",
        "resampling_applied": resampled,
        "source_sample_rate_hz": int(source_rate),
        "target_sample_rate_hz": MODEL_SAMPLE_RATE,
        "resampler": "soxr HQ" if resampled else None,
        "non_finite_sample_behavior": "reject the complete WAV before resampling",
    }
    return AudioData(
        samples=converted,
        sample_rate=MODEL_SAMPLE_RATE,
        original_wav_metadata=metadata,
        transform=transform,
    )


def window_bounds(sample_count: int, sample_rate: int) -> list[tuple[int, int]]:
    """Generate three-second/1.5-second windows and retain the final partial window."""
    if sample_count <= 0 or sample_rate <= 0:
        return []
    duration = sample_count / sample_rate
    if duration <= WINDOW_SECONDS:
        starts = [0.0]
    else:
        last_start_count = math.ceil(
            max(0.0, duration - WINDOW_SECONDS) / WINDOW_STRIDE_SECONDS - 1e-12
        )
        starts = [index * WINDOW_STRIDE_SECONDS for index in range(last_start_count + 1)]
    bounds = []
    for start in starts:
        start_index = min(sample_count, int(round(start * sample_rate)))
        end_index = min(sample_count, int(round(min(start + WINDOW_SECONDS, duration) * sample_rate)))
        if end_index > start_index:
            bounds.append((start_index, end_index))
    return bounds


def _acoustic_metrics(segment: np.ndarray) -> dict[str, float]:
    values = np.asarray(segment, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(values))))
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if values.size < 2:
        return {"rms_energy": rms, "peak_amplitude": peak, "spectral_flatness": 0.0,
                "dominant_peak_fraction": 0.0, "rms_cv": 0.0}

    spectrum = np.abs(np.fft.rfft(values * np.hanning(values.size))) ** 2
    positive = spectrum[1:] if spectrum.size > 1 else spectrum
    positive = np.maximum(positive, np.finfo(np.float64).tiny)
    flatness = float(np.exp(np.mean(np.log(positive))) / np.mean(positive))
    power_sum = float(np.sum(positive))
    dominant_fraction = float(np.max(positive) / power_sum) if power_sum > 0 else 0.0

    frame_samples = max(1, int(0.05 * MODEL_SAMPLE_RATE))
    frame_rms = [
        float(np.sqrt(np.mean(np.square(values[offset:offset + frame_samples]))))
        for offset in range(0, len(values), frame_samples)
        if len(values[offset:offset + frame_samples])
    ]
    frame_mean = float(np.mean(frame_rms)) if frame_rms else 0.0
    rms_cv = float(np.std(frame_rms) / frame_mean) if frame_mean > 0 else 0.0
    return {
        "rms_energy": rms,
        "peak_amplitude": peak,
        "spectral_flatness": flatness,
        "dominant_peak_fraction": dominant_fraction,
        "rms_cv": rms_cv,
    }


def _vad_ratio(segment: np.ndarray) -> float | None:
    try:
        import webrtcvad
    except ImportError:
        return None

    frame_samples = int(MODEL_SAMPLE_RATE * 0.03)
    signal = np.clip(segment, -1.0, 1.0)
    pcm = (signal * 32767.0).astype("<i2", copy=False).tobytes()
    frames = [pcm[start:start + frame_samples * 2] for start in range(0, len(pcm), frame_samples * 2)]
    if frames and len(frames[-1]) < frame_samples * 2:
        frames[-1] = frames[-1] + b"\0" * (frame_samples * 2 - len(frames[-1]))
    if not frames:
        return 0.0
    try:
        detector = webrtcvad.Vad(2)
        activity = [detector.is_speech(frame, MODEL_SAMPLE_RATE) for frame in frames]
    except Exception:
        return None
    return float(sum(activity) / len(activity))


def screen_window(segment: np.ndarray) -> tuple[list[str], dict[str, float | None]]:
    """Return non-blocking quality flags and measured acoustic indicators."""
    metrics = _acoustic_metrics(segment)
    if metrics["rms_energy"] <= DIGITAL_SILENCE_RMS_FLOOR:
        return [], {**metrics, "vad_activity_ratio": None}
    flags: list[str] = []
    if metrics["rms_energy"] < LOW_RMS_FLAG_THRESHOLD:
        flags.append("LOW_RMS_ENERGY")
    if metrics["peak_amplitude"] >= CLIPPING_PEAK_THRESHOLD:
        flags.append("CLIPPING_DETECTED")
    if (
        metrics["spectral_flatness"] <= TONAL_SPECTRAL_FLATNESS_MAX
        or metrics["dominant_peak_fraction"] >= TONAL_DOMINANT_POWER_FRACTION
    ):
        flags.append("TONAL_SUSPECTED")
    if (
        metrics["rms_cv"] <= STATIONARY_NOISE_RMS_CV_MAX
        and metrics["spectral_flatness"] >= STATIONARY_NOISE_SPECTRAL_FLATNESS_MIN
    ):
        flags.append("STATIONARY_NOISE_SUSPECTED")

    ratio = _vad_ratio(segment)
    if ratio is None:
        flags.append("VAD_UNAVAILABLE")
    elif ratio < LOW_VAD_ACTIVITY_RATIO:
        flags.append("LOW_VAD_ACTIVITY")
    indicators: dict[str, float | None] = {**metrics, "vad_activity_ratio": ratio}
    return sorted(set(flags)), indicators


def _verified_snapshot_path(revision: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ModelVerificationError("Configured model revision is not a full immutable commit SHA")
    from huggingface_hub import snapshot_download

    patterns = ["config.json", "preprocessor_config.json", "model.safetensors"]
    try:
        path = snapshot_download(
            repo_id=MODEL_ID,
            revision=revision,
            local_files_only=True,
            allow_patterns=patterns,
        )
    except Exception as cache_error:
        try:
            path = snapshot_download(
                repo_id=MODEL_ID,
                revision=revision,
                allow_patterns=patterns,
            )
        except Exception as online_error:
            raise ModelVerificationError(
                f"Pinned model revision is neither available online nor complete in the local cache: {online_error}"
            ) from cache_error
    if path.rstrip("\\/").split("/")[-1].split("\\")[-1].lower() != revision.lower():
        raise ModelVerificationError("Hugging Face returned a cache snapshot for a different revision")
    return path


def _classifier_head_structure(model: Any) -> list[dict[str, Any]]:
    import torch

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(
            token in name.lower().split(".")
            for token in ("classifier", "projector", "score", "classification_head")
        ):
            layers.append({
                "module_name": name or "<root>",
                "type": module.__class__.__name__,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
            })
    return layers


def load_model() -> ModelBundle:
    """Load and re-verify config, safetensors weights, and feature extractor."""
    if not re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION):
        raise ModelVerificationError("Model analysis is disabled: configured revision pin is missing or invalid")
    try:
        import torch
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification
    except Exception as exc:
        raise ModelVerificationError(f"Required model packages could not be imported: {exc}") from exc

    snapshot_path = _verified_snapshot_path(MODEL_REVISION)
    common = {"revision": MODEL_REVISION, "local_files_only": True, "trust_remote_code": False}
    try:
        config = AutoConfig.from_pretrained(MODEL_ID, **common)
        extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, **common)
        model = AutoModelForAudioClassification.from_pretrained(
            MODEL_ID,
            use_safetensors=True,
            **common,
        )
    except Exception as exc:
        raise ModelVerificationError(f"Pinned model files failed to load safely: {exc}") from exc

    labels = validate_label_mapping(
        config.id2label,
        config.label2id,
        config.num_labels,
    )
    model_labels = validate_label_mapping(
        model.config.id2label,
        model.config.label2id,
        model.config.num_labels,
    )
    if labels != model_labels:
        raise ModelVerificationError("Loaded model metadata disagrees with the pinned configuration")

    loaded_hash = getattr(model.config, "_commit_hash", None)
    config_hash = getattr(config, "_commit_hash", None)
    if loaded_hash and loaded_hash.lower() != MODEL_REVISION:
        raise ModelVerificationError("Loaded weights report a different Hub revision")
    if config_hash and config_hash.lower() != MODEL_REVISION:
        raise ModelVerificationError("Loaded config reports a different Hub revision")
    extractor_hash = getattr(extractor, "_commit_hash", None)
    if extractor_hash and extractor_hash.lower() != MODEL_REVISION:
        raise ModelVerificationError("Loaded feature extractor reports a different Hub revision")
    if int(getattr(extractor, "sampling_rate", 0)) != MODEL_SAMPLE_RATE:
        raise ModelVerificationError("Feature extractor sampling rate disagrees with configuration")
    if not getattr(config, "_commit_hash", None) and not getattr(model.config, "_commit_hash", None):
        # All objects were loaded by the exact immutable revision from this verified snapshot.
        if snapshot_path.rstrip("\\/").split("/")[-1].split("\\")[-1].lower() != MODEL_REVISION:
            raise ModelVerificationError("Loaded model revision could not be confirmed")

    model.eval().to("cpu")
    synthetic = np.zeros(MODEL_SAMPLE_RATE, dtype=np.float32)
    inputs = extractor(synthetic, sampling_rate=MODEL_SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    if logits.ndim != 2 or logits.shape[0] != 1 or int(logits.shape[1]) != MODEL_OUTPUT_DIMENSION:
        raise ModelVerificationError(
            f"Synthetic forward pass produced logits shape {tuple(logits.shape)}; expected (1, {MODEL_OUTPUT_DIMENSION})"
        )
    if not torch.isfinite(logits).all().item():
        raise ModelVerificationError("Synthetic forward pass produced non-finite logits")

    verification = {
        "status": "verified",
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "loaded_revision": MODEL_REVISION,
        "snapshot_path_revision": MODEL_REVISION,
        "raw_id2label": {str(index): label for index, label in enumerate(labels)},
        "raw_label2id": {label: index for index, label in enumerate(labels)},
        "num_labels": int(config.num_labels),
        "output_dimension": int(logits.shape[-1]),
        "synthetic_forward_pass": "passed",
        "feature_extractor_sampling_rate_hz": int(extractor.sampling_rate),
        "classifier_head_linear_modules": _classifier_head_structure(model),
        "safetensors_requested": True,
        "human_approved": False,
    }
    return ModelBundle(
        model=model,
        feature_extractor=extractor,
        revision=MODEL_REVISION,
        labels=labels,
        output_dimension=int(logits.shape[-1]),
        sample_rate=MODEL_SAMPLE_RATE,
        verification=verification,
    )


def dependency_versions() -> dict[str, str | None]:
    packages = (
        "python",
        "numpy",
        "soundfile",
        "soxr",
        "torch",
        "transformers",
        "huggingface-hub",
        "safetensors",
        "streamlit",
        "webrtcvad-wheels",
    )
    import sys

    versions = {"python": sys.version.split()[0]}
    for package in packages[1:]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def analyze_audio(
    audio: AudioData,
    classifier: ModelBundle | Callable[[np.ndarray], Any],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    """Screen, classify, and account for every established window independently."""
    samples = audio.samples
    bounds = window_bounds(len(samples), audio.sample_rate)
    windows: list[dict[str, Any]] = []
    for index, (start_sample, end_sample) in enumerate(bounds):
        start_seconds = start_sample / audio.sample_rate
        end_seconds = end_sample / audio.sample_rate
        segment = samples[start_sample:end_sample]
        flags, metrics = screen_window(segment)
        row: dict[str, Any] = {
            "index": index,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "status": "pending",
            "reason": None,
            "scores": [],
            "scores_by_label": {},
            "quality_flags": flags,
            "metrics": metrics,
        }
        segment_duration = (end_sample - start_sample) / audio.sample_rate
        if segment_duration < MIN_ANALYZABLE_WINDOW_SECONDS:
            row["status"] = "skipped"
            row["reason"] = "Window is shorter than the provisional minimum analyzable duration"
        elif metrics["rms_energy"] <= DIGITAL_SILENCE_RMS_FLOOR:
            row["status"] = "skipped"
            row["reason"] = "Effectively digital silence at the provisional RMS floor"
        else:
            try:
                if isinstance(classifier, ModelBundle):
                    raw_scores = classifier.predict_scores(segment)
                    labels = classifier.labels
                    output_dimension = classifier.output_dimension
                elif hasattr(classifier, "predict_scores"):
                    raw_scores = classifier.predict_scores(segment)
                    labels = tuple(EXPECTED_ID2LABEL[index] for index in range(MODEL_OUTPUT_DIMENSION))
                    output_dimension = MODEL_OUTPUT_DIMENSION
                else:
                    raw_scores = classifier(segment)
                    labels = tuple(EXPECTED_ID2LABEL[index] for index in range(MODEL_OUTPUT_DIMENSION))
                    output_dimension = MODEL_OUTPUT_DIMENSION
                scores = validate_score_vector(raw_scores, labels, output_dimension)
                row["scores"] = scores
                row["scores_by_label"] = {item["label"]: item["score"] for item in scores}
                row["status"] = "classified"
            except Exception as exc:
                row["status"] = "failed"
                row["reason"] = f"Window classification or output validation failed: {type(exc).__name__}: {exc}"
        windows.append(row)
        if progress_callback is not None:
            progress_callback(index + 1, len(bounds))

    coverage = coverage_accounting(
        audio.original_wav_metadata["duration_seconds"],
        windows,
    )
    stats = class_statistics(
        windows,
        tuple(EXPECTED_ID2LABEL[index] for index in range(MODEL_OUTPUT_DIMENSION)),
        coverage["coverage_state"],
    )
    return windows, coverage, stats
