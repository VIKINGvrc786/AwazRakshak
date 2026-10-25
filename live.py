"""In-memory microphone buffering and background inference."""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import soxr

from config import (
    DIGITAL_SILENCE_RMS_FLOOR,
    LIVE_BUFFER_SECONDS,
    LIVE_CATEGORY_CONFIRM_WINDOWS,
    LIVE_MAX_INSIGHTS,
    LIVE_SMOOTHING_ALPHA,
    MODEL_SAMPLE_RATE,
    WINDOW_SECONDS,
)
from engine import ModelBundle, screen_window, validate_score_vector
from signals import categorize_call, derive_indicators, smooth_indicator

_LOW_CONFIDENCE_FLAGS = frozenset(
    {
        "CLIPPING_DETECTED",
        "LOW_RMS_ENERGY",
        "LOW_VAD_ACTIVITY",
        "STATIONARY_NOISE_SUSPECTED",
        "TONAL_SUSPECTED",
    }
)


class LiveAudioBuffer:
    """A bounded rolling mono buffer; audio is retained only in process memory."""

    def __init__(
        self,
        sample_rate: int = MODEL_SAMPLE_RATE,
        max_seconds: float = LIVE_BUFFER_SECONDS,
    ) -> None:
        if sample_rate <= 0 or max_seconds <= 0:
            raise ValueError("Sample rate and buffer duration must be positive")
        self.sample_rate = sample_rate
        self.capacity = int(sample_rate * max_seconds)
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._index = 0
        self._size = 0
        self._total = 0
        self._lock = threading.Lock()
        self._input_sample_rate: int | None = None
        self._resampler: soxr.ResampleStream | None = None

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total

    def append(self, samples: np.ndarray, sample_rate: int) -> int:
        if sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive")
        audio = np.asarray(samples)
        if np.issubdtype(audio.dtype, np.integer):
            limits = np.iinfo(audio.dtype)
            scale = float(max(abs(limits.min), limits.max))
            audio = audio.astype(np.float32) / scale
        if audio.ndim == 2:
            channel_axis = 0 if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1] else 1
            audio = audio.mean(axis=channel_axis)
        if audio.ndim != 1 or not audio.size:
            raise ValueError("Audio frame must contain one or more samples")

        if not np.issubdtype(audio.dtype, np.floating):
            audio = audio.astype(np.float32, copy=False)
        if not np.isfinite(audio).all():
            raise ValueError("Audio frame contains non-finite samples")

        if sample_rate != self.sample_rate:
            if sample_rate != self._input_sample_rate:
                self.flush()
                self._input_sample_rate = sample_rate
                self._resampler = soxr.ResampleStream(
                    sample_rate,
                    self.sample_rate,
                    num_channels=1,
                    dtype="float32",
                )
            audio = self._resampler.resample_chunk(audio, last=False)
        elif self._resampler is not None:
            self.flush()
        if len(audio) >= self.capacity:
            audio = audio[-self.capacity :]

        self._write(audio)
        return len(audio)

    def flush(self) -> None:
        if self._resampler is None:
            return
        remaining = self._resampler.resample_chunk(
            np.empty(0, dtype=np.float32),
            last=True,
        )
        self._resampler = None
        self._input_sample_rate = None
        if remaining.size:
            self._write(remaining)

    def _write(self, audio: np.ndarray) -> None:
        with self._lock:
            first_count = min(len(audio), self.capacity - self._index)
            self._data[self._index : self._index + first_count] = audio[:first_count]
            remaining = len(audio) - first_count
            if remaining:
                self._data[:remaining] = audio[first_count:]
            self._index = (self._index + len(audio)) % self.capacity
            self._size = min(self.capacity, self._size + len(audio))
            self._total += len(audio)

    def snapshot(self, seconds: float | None = None) -> np.ndarray:
        with self._lock:
            count = (
                self._size
                if seconds is None
                else min(self._size, int(seconds * self.sample_rate))
            )
            start = (self._index - count) % self.capacity
            if start + count <= self.capacity:
                return self._data[start : start + count].copy()
            first_count = self.capacity - start
            return np.concatenate((self._data[start:], self._data[: count - first_count]))

    def clear(self) -> None:
        with self._lock:
            self._data.fill(0.0)
            self._index = 0
            self._size = 0
            self._total = 0
            self._resampler = None
            self._input_sample_rate = None


class LiveCallPipeline:
    """Buffer microphone frames and analyze recent windows off the UI thread."""

    def __init__(
        self,
        model_factory: Callable[[], ModelBundle],
        update_interval: float = 1.5,
    ) -> None:
        if update_interval <= 0:
            raise ValueError("Update interval must be positive")
        self.buffer = LiveAudioBuffer()
        self.model_factory = model_factory
        self.interval = update_interval
        self._lock = threading.Lock()
        self._ingest_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._stop.set()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] | None = None
        self._error: str | None = None
        self._insights: deque[dict[str, Any]] = deque(maxlen=LIVE_MAX_INSIGHTS)
        self._events: deque[dict[str, Any]] = deque(maxlen=LIVE_MAX_INSIGHTS)
        self._recent_quality: deque[bool] = deque(maxlen=3)
        self._candidate_history: list[str] = []
        self._category = "Routine"
        self._smooth: dict[str, float] = {}
        self._last_sample_count = 0
        self._started_monotonic = 0.0
        self._stopped_monotonic: float | None = None
        self._started_at: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.buffer.clear()
        self._wake.clear()
        self._stop.clear()
        self._latest = None
        self._error = None
        self._insights.clear()
        self._events.clear()
        self._recent_quality.clear()
        self._candidate_history.clear()
        self._category = "Routine"
        self._smooth.clear()
        self._last_sample_count = 0
        self._started_monotonic = time.monotonic()
        self._stopped_monotonic = None
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._thread = threading.Thread(
            target=self._worker,
            name="awazrakshak-live-inference",
            daemon=True,
        )
        self._thread.start()

    def add_audio(self, samples: np.ndarray, sample_rate: int) -> int:
        with self._ingest_lock:
            if self._stop.is_set():
                return 0
            count = self.buffer.append(samples, sample_rate)
            self._wake.set()
            return count

    def stop(self) -> None:
        if self._stopped_monotonic is None and self._started_monotonic:
            self._stopped_monotonic = time.monotonic()
        self._stop.set()
        self._wake.set()
        with self._ingest_lock:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.buffer.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            end = self._stopped_monotonic or time.monotonic()
            duration = (
                max(0.0, end - self._started_monotonic)
                if self._started_monotonic
                else 0.0
            )
            return {
                "latest": copy.deepcopy(self._latest),
                "insights": copy.deepcopy(list(self._insights)),
                "events": copy.deepcopy(list(self._events)),
                "error": self._error,
                "recording": not self._stop.is_set(),
                "started_at": self._started_at,
                "duration_seconds": duration,
            }

    def _worker(self) -> None:
        try:
            model = self.model_factory()
        except Exception as exc:
            with self._lock:
                self._error = f"Analysis disabled: {exc}"
            return

        required_samples = int(WINDOW_SECONDS * MODEL_SAMPLE_RATE)
        stride_samples = max(1, int(self.interval * MODEL_SAMPLE_RATE))
        while not self._stop.is_set():
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            if self._stop.is_set():
                break
            available_samples = self.buffer.total_samples
            if (
                available_samples - self._last_sample_count < stride_samples
                or available_samples < required_samples
            ):
                continue

            window = self.buffer.snapshot(WINDOW_SECONDS)
            self._last_sample_count = available_samples
            try:
                flags, metrics = screen_window(window)
                if float(metrics["rms_energy"]) <= DIGITAL_SILENCE_RMS_FLOOR:
                    continue
                scores = validate_score_vector(
                    model.predict_scores(window),
                    model.labels,
                    model.output_dimension,
                )
                raw_indicators = derive_indicators(
                    {item["label"]: float(item["score"]) for item in scores},
                    metrics,
                )
                for name in ("distress_indicator", "nervousness_indicator"):
                    history = (
                        [self._smooth[name], raw_indicators[name]]
                        if name in self._smooth
                        else [raw_indicators[name]]
                    )
                    self._smooth[name] = smooth_indicator(
                        history,
                        LIVE_SMOOTHING_ALPHA,
                    )
                previous_category = self._category
                proposed_category = categorize_call(
                    self._smooth,
                    flags,
                    previous_category=previous_category,
                    candidate_history=self._candidate_history,
                )
                if proposed_category != previous_category:
                    self._category = proposed_category
                    self._candidate_history.clear()
                    event = {
                        "time": time.monotonic() - self._started_monotonic,
                        "from": previous_category,
                        "to": proposed_category,
                    }
                else:
                    event = None
                    candidate = categorize_call(self._smooth, flags)
                    if candidate == self._category:
                        self._candidate_history.clear()
                    else:
                        if (
                            self._candidate_history
                            and self._candidate_history[-1] != candidate
                        ):
                            self._candidate_history.clear()
                        self._candidate_history.append(candidate)
                        self._candidate_history = self._candidate_history[
                            -(LIVE_CATEGORY_CONFIRM_WINDOWS - 1) :
                        ]

                self._recent_quality.append(
                    any(flag in _LOW_CONFIDENCE_FLAGS for flag in flags)
                )
                insight = {
                    "elapsed": time.monotonic() - self._started_monotonic,
                    "scores": scores,
                    "indicators": dict(self._smooth),
                    "quality_flags": flags,
                    "low_confidence": any(self._recent_quality),
                    "category": self._category,
                }
                with self._lock:
                    if self._stop.is_set():
                        break
                    if event is not None:
                        self._events.append(event)
                    self._latest = insight
                    self._insights.append(insight)
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = f"Analysis disabled: {exc}"
