"""Strict, audio-free JSON analysis receipts and fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

RECEIPT_VERSION = "3.0"

LIMITATIONS = [
    "Experimental investigator-support prototype; human listening and review are required.",
    "Raw model scores are not calibrated probabilities or proof of a real emotional state.",
    "This system does not determine danger, urgency, deception, guilt, fraud, intent, or mental-health status.",
    "Nervousness and distress are not independent classes in the pinned checkpoint.",
    "Low scores and missing predictions are not evidence that audio is safe, calm, or free of emotion.",
    "The closed-set classifier produces outputs for non-speech and unrelated sounds; noise can receive a high score.",
    "Coverage states measure audio processing coverage only and do not validate predictions or establish safety.",
    "Input fingerprints prove byte equality only, not recording authenticity or speaker identity.",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_sha256(data: bytes, expected: str) -> bool:
    candidate = str(expected).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", candidate):
        return False
    return sha256_bytes(data) == candidate


def build_receipt(
    *,
    input_bytes: bytes,
    original_wav_metadata: Mapping[str, Any],
    audio_transform: Mapping[str, Any],
    model: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    class_statistics: Mapping[str, Any] | None,
    dependency_versions: Mapping[str, Any],
    provisional_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Build version 3.0 receipt without retaining uploaded audio or arrays."""
    return {
        "schema_version": RECEIPT_VERSION,
        "application": "VoxTrace",
        "input": {
            "sha256": sha256_bytes(input_bytes),
            "file_size_bytes": len(input_bytes),
            "original_wav_metadata": dict(original_wav_metadata),
            "fingerprint_scope": "SHA-256 of the original uploaded bytes",
            "fingerprint_limitation": (
                "A matching fingerprint proves byte equality against the supplied reference; "
                "it does not prove recording authenticity or speaker identity."
            ),
        },
        "audio_transform": dict(audio_transform),
        "model": dict(model),
        "windows": [dict(window) for window in windows],
        "coverage": dict(coverage),
        "class_statistics": dict(class_statistics) if class_statistics is not None else None,
        "dependency_versions": dict(dependency_versions),
        "provisional_thresholds": dict(provisional_thresholds),
        "limitations": list(LIMITATIONS),
    }


def strict_json_dumps(receipt: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize standards-compliant JSON; reject NaN and infinity."""
    return json.dumps(receipt, indent=indent, ensure_ascii=False, allow_nan=False)
