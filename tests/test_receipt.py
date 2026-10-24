from __future__ import annotations

import json

import pytest

from receipt import RECEIPT_VERSION, build_receipt, sha256_bytes, strict_json_dumps, verify_sha256


def sample_receipt(input_bytes=b"wav bytes"):
    return build_receipt(
        input_bytes=input_bytes,
        original_wav_metadata={"format": "WAV", "sample_rate_hz": 16_000, "channels": 1, "frames": 48_000, "duration_seconds": 3.0},
        audio_transform={"target_sample_rate_hz": 16_000, "resampling_applied": False},
        model={"model_id": "Dpngtm/wav2vec2-emotion-recognition", "requested_revision": "a" * 40,
               "loaded_revision": "a" * 40, "verification_status": "verified",
               "raw_id2label": {"0": "angry", "1": "calm"}, "output_dimension": 7},
        windows=[{"index": 0, "start_seconds": 0.0, "end_seconds": 3.0, "status": "classified",
                  "scores": [{"label": "angry", "score": 1.0}], "quality_flags": []}],
        coverage={"coverage_state": "PARTIAL_COVERAGE", "classified_all_seconds": 3.0},
        class_statistics=None,
        dependency_versions={"python": "3.12.10"},
        provisional_thresholds={"status": "provisional"},
    )


def test_receipt_round_trip_is_strict_and_audio_free():
    receipt = sample_receipt()
    encoded = strict_json_dumps(receipt)
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == RECEIPT_VERSION == "3.0"
    assert decoded["input"]["sha256"] == sha256_bytes(b"wav bytes")
    assert decoded["model"]["requested_revision"] == "a" * 40
    assert decoded["model"]["raw_id2label"]["1"] == "calm"
    assert "audio_bytes" not in decoded["input"]
    assert "samples" not in encoded
    assert "wav bytes" not in encoded


def test_fingerprint_matches_original_bytes_and_detects_changed_bytes():
    fingerprint = sha256_bytes(b"original")
    assert verify_sha256(b"original", fingerprint)
    assert not verify_sha256(b"changed", fingerprint)
    assert not verify_sha256(b"original", "not-a-hash")


def test_strict_json_rejects_non_finite_values():
    receipt = sample_receipt()
    receipt["coverage"]["invalid"] = float("nan")
    with pytest.raises(ValueError):
        strict_json_dumps(receipt)
