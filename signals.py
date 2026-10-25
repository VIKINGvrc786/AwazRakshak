"""Derived voice indicators and attention-level helpers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from config import (
    PROVISIONAL_ATTENTION_THRESHOLDS,
    PROVISIONAL_INDICATOR_WEIGHTS,
    LIVE_CATEGORY_CONFIRM_WINDOWS,
    LIVE_CATEGORY_THRESHOLDS,
)


def derive_indicators(
    scores_by_label: Mapping[str, float],
    metrics: Mapping[str, float],
) -> dict[str, float]:
    """Calculate provisional indicators without adding model classes."""
    required_labels = {
        label
        for weights in PROVISIONAL_INDICATOR_WEIGHTS.values()
        for label in weights
        if label != "rms_cv"
    }
    if not required_labels.issubset(scores_by_label):
        missing = sorted(required_labels - set(scores_by_label))
        raise ValueError(f"Missing raw scores required for derived indicators: {missing}")

    for label in required_labels:
        score = float(scores_by_label[label])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Raw score for {label!r} must be finite and within [0, 1]")

    if "rms_cv" not in metrics:
        raise ValueError("Acoustic metrics must include rms_cv")
    rms_cv = float(metrics["rms_cv"])
    if not math.isfinite(rms_cv) or rms_cv < 0.0:
        raise ValueError("rms_cv must be finite and non-negative")

    distress_weights = PROVISIONAL_INDICATOR_WEIGHTS["distress"]
    nervousness_weights = PROVISIONAL_INDICATOR_WEIGHTS["nervousness"]
    return {
        "distress_indicator": sum(
            weight * float(scores_by_label[label])
            for label, weight in distress_weights.items()
        ),
        "nervousness_indicator": sum(
            weight * (
                min(rms_cv, 1.0)
                if label == "rms_cv"
                else float(scores_by_label[label])
            )
            for label, weight in nervousness_weights.items()
        ),
        "fear": float(scores_by_label["fearful"]),
        "anger": float(scores_by_label["angry"]),
    }


def attention_level(indicator: float) -> str:
    """Map a provisional indicator to a neutral display category."""
    value = float(indicator)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Indicator must be finite and within [0, 1]")
    if value < PROVISIONAL_ATTENTION_THRESHOLDS["elevated"]:
        return "Low"
    if value < PROVISIONAL_ATTENTION_THRESHOLDS["high"]:
        return "Elevated"
    return "High"


def smooth_indicator(history: Sequence[float], alpha: float) -> float:
    """Return the latest exponential moving average from a non-empty history."""
    smoothing = float(alpha)
    if not math.isfinite(smoothing) or not 0.0 < smoothing <= 1.0:
        raise ValueError("alpha must be finite and within (0, 1]")
    if not history:
        raise ValueError("history must contain at least one indicator")
    values = [float(value) for value in history]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("indicator history values must be finite and within [0, 1]")
    smoothed = values[0]
    for value in values[1:]:
        smoothed = smoothing * value + (1.0 - smoothing) * smoothed
    return smoothed


def categorize_call(
    smoothed_indicators: Mapping[str, float],
    flags: Sequence[str],
    *,
    previous_category: str | None = None,
    candidate_history: Sequence[str] = (),
) -> str:
    """Categorize provisional indicators with two-window hysteresis."""
    required = ("distress_indicator", "nervousness_indicator")
    if any(name not in smoothed_indicators for name in required):
        raise ValueError("Both smoothed indicators are required")
    values = [float(smoothed_indicators[name]) for name in required]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Smoothed indicators must be finite and within [0, 1]")

    peak = max(values)
    if peak >= LIVE_CATEGORY_THRESHOLDS["high"]:
        candidate = "High"
    elif peak >= LIVE_CATEGORY_THRESHOLDS["elevated"]:
        candidate = "Elevated"
    else:
        candidate = "Routine"
    if previous_category not in {"Routine", "Elevated", "High"}:
        return candidate
    if candidate == previous_category:
        return previous_category
    if len(candidate_history) < LIVE_CATEGORY_CONFIRM_WINDOWS - 1:
        return previous_category
    recent = [*candidate_history, candidate][-LIVE_CATEGORY_CONFIRM_WINDOWS:]
    return candidate if all(item == candidate for item in recent) else previous_category














