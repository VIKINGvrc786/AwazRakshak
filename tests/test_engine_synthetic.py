from __future__ import annotations

import io
import sys

import numpy as np
import pytest
import soundfile as sf

from config import (
    EXPECTED_ID2LABEL,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    TONAL_DOMINANT_POWER_FRACTION,
    TONAL_SPECTRAL_FLATNESS_MAX,
)
from engine import (
    AudioData,
    AudioValidationError,
    ModelVerificationError,
    _verified_snapshot_path,
    analyze_audio,
    load_audio,
    screen_window,
    validate_label_mapping,
    validate_score_vector,
    window_bounds,
)


LABELS = tuple(EXPECTED_ID2LABEL[index] for index in sorted(EXPECTED_ID2LABEL))
GOOD_SCORES = [{"label": label, "score": 1.0 / 7} for label in LABELS]


def make_wav(samples: np.ndarray, sample_rate: int, subtype="PCM_16") -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


def make_audio(samples: np.ndarray, sample_rate=MODEL_SAMPLE_RATE) -> AudioData:
    return AudioData(
        samples=np.asarray(samples, dtype=np.float32),
        sample_rate=sample_rate,
        original_wav_metadata={
            "format": "WAV",
            "subtype": "PCM_16",
            "sample_rate_hz": sample_rate,
            "channels": 1,
            "frames": len(samples),
            "duration_seconds": len(samples) / sample_rate,
        },
        transform={"resampling_applied": False},
    )


def test_verified_revision_is_a_full_immutable_sha():
    assert len(MODEL_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in MODEL_REVISION)


def test_label_mapping_and_output_dimension_match_observed_checkpoint():
    raw = {str(index): label for index, label in EXPECTED_ID2LABEL.items()}
    reverse = {label: index for index, label in EXPECTED_ID2LABEL.items()}
    assert validate_label_mapping(raw, reverse, 7) == LABELS
    with pytest.raises(ModelVerificationError):
        validate_label_mapping(raw, reverse, 6)


@pytest.mark.parametrize("legacy", ["fear", "neutral", "anger", "sadness", "surprise"])
def test_label_map_rejects_legacy_or_substituted_names(legacy):
    mapping = dict(EXPECTED_ID2LABEL)
    original_to_legacy = {
        "fearful": "fear",
        "calm": "neutral",
        "angry": "anger",
        "sad": "sadness",
        "surprised": "surprise",
    }
    target_id = next(index for index, name in mapping.items() if original_to_legacy.get(name) == legacy)
    old_name = mapping[target_id]
    mapping[target_id] = legacy
    reverse = {name: index for index, name in mapping.items()}
    with pytest.raises(ModelVerificationError):
        validate_label_mapping(mapping, reverse, 7)
    mapping[target_id] = old_name


def test_score_vector_requires_complete_unique_finite_softmax_outputs():
    assert validate_score_vector(GOOD_SCORES, LABELS, 7) == GOOD_SCORES
    with pytest.raises(ValueError, match="class count"):
        validate_score_vector(GOOD_SCORES[:-1], LABELS, 7)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_score_vector(GOOD_SCORES[:-1] + [GOOD_SCORES[0]], LABELS, 7)
    with pytest.raises(ValueError, match="Unexpected"):
        validate_score_vector(GOOD_SCORES[:-1] + [{"label": "other", "score": 1 / 7}], LABELS, 7)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, 1.01])
def test_score_vector_rejects_nan_infinity_and_out_of_range_values(bad):
    values = [dict(item) for item in GOOD_SCORES]
    values[0]["score"] = bad
    with pytest.raises(ValueError):
        validate_score_vector(values, LABELS, 7)


def test_score_vector_rejects_invalid_sum_and_dimension():
    values = [dict(item) for item in GOOD_SCORES]
    values[0]["score"] = 0.5
    with pytest.raises(ValueError, match="sum"):
        validate_score_vector(values, LABELS, 7)
    with pytest.raises(ValueError, match="dimension"):
        validate_score_vector(GOOD_SCORES, LABELS, 6)


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 44_100, 48_000])
def test_wav_ingestion_resamples_supported_pcm_rates(sample_rate):
    count = sample_rate
    t = np.arange(count, dtype=np.float32) / sample_rate
    encoded = make_wav((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)
    audio = load_audio(encoded)
    assert audio.sample_rate == MODEL_SAMPLE_RATE
    assert abs(len(audio.samples) - MODEL_SAMPLE_RATE) <= 1
    assert audio.original_wav_metadata["sample_rate_hz"] == sample_rate
    assert audio.transform["target_sample_rate_hz"] == MODEL_SAMPLE_RATE


def test_wav_ingestion_preserves_stereo_metadata_and_reduces_to_mono():
    t = np.arange(48_000, dtype=np.float32) / 48_000
    stereo = np.column_stack((0.1 * np.sin(2 * np.pi * 200 * t), 0.1 * np.sin(2 * np.pi * 400 * t)))
    audio = load_audio(make_wav(stereo.astype(np.float32), 48_000))
    assert audio.samples.ndim == 1
    assert audio.original_wav_metadata["channels"] == 2
    assert audio.transform["channel_reduction"] == "arithmetic mean to mono"


def test_float_wav_supported():
    t = np.arange(16_000, dtype=np.float32) / 16_000
    audio = load_audio(make_wav((0.04 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16_000, "FLOAT"))
    assert audio.original_wav_metadata["subtype"] == "FLOAT"


@pytest.mark.parametrize("payload", [b"", b"not a wav", b"RIFF\x24\x00\x00\x00WAVEfmt "])
def test_empty_corrupt_and_truncated_wavs_are_rejected(payload):
    with pytest.raises(AudioValidationError):
        load_audio(payload)


def test_unusably_short_wav_is_rejected():
    short = np.ones(4_000, dtype=np.float32) * 0.1
    with pytest.raises(AudioValidationError, match="too short"):
        load_audio(make_wav(short, 8_000))


def test_non_finite_audio_is_rejected_before_resampling():
    samples = np.zeros(16_000, dtype=np.float32)
    samples[200] = np.nan
    encoded = make_wav(samples, 16_000, "FLOAT")
    with pytest.raises(AudioValidationError, match="non-finite"):
        load_audio(encoded)


def test_window_boundaries_retain_tail_without_duplicate_starts():
    assert window_bounds(int(4.6 * MODEL_SAMPLE_RATE), MODEL_SAMPLE_RATE) == [
        (0, 48_000),
        (24_000, 72_000),
        (48_000, int(4.6 * MODEL_SAMPLE_RATE)),
    ]
    exact = window_bounds(int(4.5 * MODEL_SAMPLE_RATE), MODEL_SAMPLE_RATE)
    assert len(exact) == 2
    assert exact[-1][1] == int(4.5 * MODEL_SAMPLE_RATE)


def test_exact_silence_is_skipped_not_classified():
    audio = make_audio(np.zeros(3 * MODEL_SAMPLE_RATE, dtype=np.float32))
    windows, coverage, stats = analyze_audio(audio, lambda _: GOOD_SCORES)
    assert len(windows) == 1
    assert windows[0]["status"] == "skipped"
    assert windows[0]["reason"].startswith("Effectively digital silence")
    assert coverage["coverage_state"] == "NO_ANALYZED_AUDIO"
    assert stats is None


@pytest.mark.parametrize("kind", ["tone", "white", "brown"])
def test_non_speech_tone_and_noise_are_flagged_but_still_classified(kind):
    rng = np.random.default_rng(23)
    count = 3 * MODEL_SAMPLE_RATE
    if kind == "tone":
        t = np.arange(count) / MODEL_SAMPLE_RATE
        samples = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    elif kind == "white":
        samples = (rng.normal(size=count) * 0.05).astype(np.float32)
    else:
        samples = np.cumsum(rng.normal(size=count)).astype(np.float32)
        samples -= samples.mean()
        samples *= 0.05 / max(float(np.max(np.abs(samples))), 1e-6)
    audio = make_audio(samples)
    windows, _, _ = analyze_audio(audio, lambda _: GOOD_SCORES)
    assert windows[0]["status"] == "classified"
    assert windows[0]["quality_flags"]
    if kind == "tone":
        assert "TONAL_SUSPECTED" in windows[0]["quality_flags"]


def test_harmonic_voice_like_signal_can_trigger_tonal_flag_and_remains_classified():
    t = np.arange(3 * MODEL_SAMPLE_RATE, dtype=np.float64) / MODEL_SAMPLE_RATE
    samples = sum(
        (0.08 / harmonic) * np.sin(2 * np.pi * 120 * harmonic * t)
        for harmonic in range(1, 9)
    ).astype(np.float32)

    windows, _, _ = analyze_audio(make_audio(samples), lambda _: GOOD_SCORES)

    assert windows[0]["status"] == "classified"
    assert "TONAL_SUSPECTED" in windows[0]["quality_flags"]
    metrics = windows[0]["metrics"]
    assert (
        metrics["spectral_flatness"] <= TONAL_SPECTRAL_FLATNESS_MAX
        or metrics["dominant_peak_fraction"] >= TONAL_DOMINANT_POWER_FRACTION
    )


def test_low_energy_and_clipping_flags_do_not_hard_skip_usable_audio():
    quiet = np.full(3 * MODEL_SAMPLE_RATE, 0.001, dtype=np.float32)
    quiet_windows, _, _ = analyze_audio(make_audio(quiet), lambda _: GOOD_SCORES)
    assert quiet_windows[0]["status"] == "classified"
    assert "LOW_RMS_ENERGY" in quiet_windows[0]["quality_flags"]

    clipped = np.full(3 * MODEL_SAMPLE_RATE, 0.005, dtype=np.float32)
    clipped[100] = 1.0
    clipped_windows, _, _ = analyze_audio(make_audio(clipped), lambda _: GOOD_SCORES)
    assert clipped_windows[0]["status"] == "classified"
    assert "CLIPPING_DETECTED" in clipped_windows[0]["quality_flags"]


def test_missing_optional_vad_flags_audio_but_does_not_skip(monkeypatch):
    monkeypatch.setitem(sys.modules, "webrtcvad", None)
    samples = np.full(3 * MODEL_SAMPLE_RATE, 0.03, dtype=np.float32)
    windows, _, _ = analyze_audio(make_audio(samples), lambda _: GOOD_SCORES)
    assert windows[0]["status"] == "classified"
    assert "VAD_UNAVAILABLE" in windows[0]["quality_flags"]


def test_one_window_inference_failure_does_not_crash_other_windows():
    audio = make_audio(np.full(int(4.5 * MODEL_SAMPLE_RATE), 0.04, dtype=np.float32))
    calls = 0

    def classifier(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic failure")
        return GOOD_SCORES

    windows, coverage, stats = analyze_audio(audio, classifier)
    assert [window["status"] for window in windows] == ["classified", "failed"]
    assert coverage["successfully_classified_windows"] == 1
    assert coverage["failed_windows"] == 1
    assert stats is None


def test_thirty_windows_keep_two_successes_and_account_for_all_others():
    audio = make_audio(np.full(int(46.5 * MODEL_SAMPLE_RATE), 0.04, dtype=np.float32))
    calls = 0

    def classifier(_):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return GOOD_SCORES
        raise RuntimeError("synthetic window error")

    windows, coverage, stats = analyze_audio(audio, classifier)
    assert len(windows) == 30
    assert coverage["successfully_classified_windows"] == 2
    assert coverage["failed_windows"] == 28
    assert coverage["coverage_state"] == "INSUFFICIENT_COVERAGE"
    assert stats is None


def test_all_window_failures_produce_no_analysis_state():
    audio = make_audio(np.full(3 * MODEL_SAMPLE_RATE, 0.04, dtype=np.float32))

    def fail(_):
        raise RuntimeError("expected")

    windows, coverage, stats = analyze_audio(audio, fail)
    assert windows[0]["status"] == "failed"
    assert coverage["coverage_state"] == "NO_ANALYZED_AUDIO"
    assert stats is None


def test_empty_or_wrong_revision_cache_fails_closed(monkeypatch):
    monkeypatch.setattr("engine._verified_snapshot_path", lambda _: (_ for _ in ()).throw(ModelVerificationError("empty cache")))
    from engine import load_model

    with pytest.raises(ModelVerificationError, match="empty cache"):
        load_model()
