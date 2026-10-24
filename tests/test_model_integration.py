from __future__ import annotations

import os

import numpy as np
import pytest

from config import EXPECTED_ID2LABEL, MODEL_REVISION, MODEL_SAMPLE_RATE
from engine import AudioData, analyze_audio, load_model
from receipt import build_receipt, strict_json_dumps


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.environ.get("VOXTRACE_RUN_REAL_MODEL_TESTS") != "1",
        reason="Set VOXTRACE_RUN_REAL_MODEL_TESTS=1 to run cached real-model integration",
    ),
]


def test_cached_real_model_inference_receipt_and_offline_after_cache():
    bundle = load_model()
    assert bundle.revision == MODEL_REVISION
    assert bundle.labels == tuple(EXPECTED_ID2LABEL[index] for index in sorted(EXPECTED_ID2LABEL))
    assert bundle.output_dimension == 7

    count = int(4.0 * MODEL_SAMPLE_RATE)
    t = np.arange(count, dtype=np.float32) / MODEL_SAMPLE_RATE
    samples = (0.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = AudioData(
        samples=samples,
        sample_rate=MODEL_SAMPLE_RATE,
        original_wav_metadata={
            "format": "WAV", "subtype": "FLOAT", "sample_rate_hz": MODEL_SAMPLE_RATE,
            "channels": 1, "frames": count, "duration_seconds": 4.0,
        },
        transform={"resampling_applied": False, "target_sample_rate_hz": MODEL_SAMPLE_RATE},
    )
    windows, coverage, stats = analyze_audio(audio, bundle)
    assert len(windows) == 2
    assert all(window["status"] == "classified" for window in windows)
    assert all(len(window["scores"]) == 7 for window in windows)
    assert coverage["coverage_state"] == "PARTIAL_COVERAGE"
    assert stats is None

    receipt = build_receipt(
        input_bytes=b"synthetic receipt fingerprint only",
        original_wav_metadata=audio.original_wav_metadata,
        audio_transform=audio.transform,
        model={
            "model_id": bundle.verification["model_id"],
            "requested_revision": bundle.verification["requested_revision"],
            "loaded_revision": bundle.revision,
            "verification_status": "verified",
            "raw_id2label": bundle.verification["raw_id2label"],
            "output_dimension": bundle.output_dimension,
        },
        windows=windows,
        coverage=coverage,
        class_statistics=stats,
        dependency_versions={"torch": "actual installed version"},
        provisional_thresholds={"status": "provisional"},
    )
    decoded = strict_json_dumps(receipt)
    assert '"loaded_revision": "' + MODEL_REVISION + '"' in decoded
    assert '"status": "classified"' in decoded
