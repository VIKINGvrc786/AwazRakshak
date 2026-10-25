import time

import numpy as np
import pytest

from config import EXPECTED_ID2LABEL
import live
from live import LiveAudioBuffer, LiveCallPipeline


class FakeModel:
    labels = tuple(EXPECTED_ID2LABEL.values())
    output_dimension = len(labels)

    def predict_scores(self, waveform):
        scores = [0.01] * self.output_dimension
        scores[3] = 0.94
        return [
            {"label": label, "score": score}
            for label, score in zip(self.labels, scores, strict=True)
        ]


def test_ring_buffer_keeps_only_latest_resampled_audio():
    buffer = LiveAudioBuffer(sample_rate=10, max_seconds=1)
    buffer.append(np.arange(8, dtype=np.float32), 10)
    assert buffer.snapshot().tolist() == list(map(float, range(8)))

    buffer.append(np.arange(8, 15, dtype=np.float32), 10)
    assert buffer.snapshot().tolist() == list(map(float, range(5, 15)))


def test_buffer_converts_integer_stereo_and_resamples_to_mono():
    buffer = LiveAudioBuffer(sample_rate=16_000, max_seconds=2)
    stereo = np.full((2, 4_000), 16_384, dtype=np.int16)

    buffer.append(stereo, 8_000)
    buffer.flush()

    assert len(buffer.snapshot()) == 8_000
    assert np.median(buffer.snapshot()) == pytest.approx(0.5, abs=0.01)


def test_pipeline_runs_inference_and_discards_audio_on_stop():
    pipeline = LiveCallPipeline(FakeModel, update_interval=0.01)
    pipeline.start()
    waveform = np.sin(
        np.linspace(0, 2 * np.pi * 220 * 3, 48_000, endpoint=False)
    ).astype(np.float32) * 0.1
    pipeline.add_audio(waveform, 16_000)

    deadline = time.monotonic() + 3
    while pipeline.snapshot()["latest"] is None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert pipeline.snapshot()["latest"] is not None
    pipeline.stop()
    snapshot = pipeline.snapshot()
    assert not snapshot["recording"]
    assert snapshot["duration_seconds"] >= 0
    assert pipeline.buffer.total_samples == 0


def test_pipeline_marks_low_quality_windows_low_confidence(monkeypatch):
    monkeypatch.setattr(
        live,
        "screen_window",
        lambda _window: (
            ["TONAL_SUSPECTED"],
            {"rms_energy": 0.1, "rms_cv": 0.2},
        ),
    )
    pipeline = LiveCallPipeline(FakeModel, update_interval=0.01)
    pipeline.start()
    pipeline.add_audio(np.ones(48_000, dtype=np.float32) * 0.1, 16_000)

    deadline = time.monotonic() + 3
    while pipeline.snapshot()["latest"] is None and time.monotonic() < deadline:
        time.sleep(0.01)

    latest = pipeline.snapshot()["latest"]
    pipeline.stop()
    assert latest is not None
    assert latest["low_confidence"]
