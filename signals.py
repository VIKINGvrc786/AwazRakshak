"""Pure time-coverage, score-statistics, and candidate-ranking helpers."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

from config import (
    MIN_ADEQUATE_COVERAGE_FRACTION,
    MIN_INSUFFICIENT_COVERAGE_FRACTION,
    MIN_SUMMARY_ANALYZED_SECONDS,
)


def merge_intervals(intervals: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    """Return the sorted union of finite, positive-length intervals.

    Adjacent intervals are merged. Invalid, non-finite, reversed, or zero-length
    intervals are rejected so coverage cannot be inflated silently.
    """
    normalized: list[tuple[float, float]] = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("Each interval must contain exactly start and end")
        start, end = float(interval[0]), float(interval[1])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Interval bounds must be finite")
        if start < 0 or end <= start:
            raise ValueError("Intervals must have start >= 0 and end > start")
        normalized.append((start, end))

    normalized.sort()
    merged: list[tuple[float, float]] = []
    for start, end in normalized:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_union_seconds(intervals: Iterable[Sequence[float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def coverage_accounting(recording_duration: float, windows: Sequence[Mapping]) -> dict:
    """Account for classified time using interval unions, never summed overlap."""
    duration = float(recording_duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Recording duration must be finite and positive")

    classified = [window for window in windows if window.get("status") == "classified"]
    flagged = [window for window in classified if window.get("quality_flags")]
    unflagged = [window for window in classified if not window.get("quality_flags")]
    skipped = [window for window in windows if window.get("status") == "skipped"]
    failed = [window for window in windows if window.get("status") == "failed"]
    completed_statuses = {"classified", "skipped", "failed"}
    completed = [window for window in windows if window.get("status") in completed_statuses]
    unprocessed = [window for window in windows if window.get("status") not in completed_statuses]

    def clipped_intervals(items: Sequence[Mapping]) -> list[tuple[float, float]]:
        intervals = []
        for item in items:
            start = max(0.0, min(duration, float(item["start_seconds"])))
            end = max(0.0, min(duration, float(item["end_seconds"])))
            if end > start:
                intervals.append((start, end))
        return intervals

    classified_seconds = min(duration, interval_union_seconds(clipped_intervals(classified)))
    unflagged_seconds = min(duration, interval_union_seconds(clipped_intervals(unflagged)))
    skipped_seconds = min(duration, interval_union_seconds(clipped_intervals(skipped)))
    failed_seconds = min(duration, interval_union_seconds(clipped_intervals(failed)))
    completed_seconds = min(duration, interval_union_seconds(clipped_intervals(completed)))
    analyzed_fraction = min(1.0, classified_seconds / duration)

    if not classified:
        state = "NO_ANALYZED_AUDIO"
    elif duration >= MIN_SUMMARY_ANALYZED_SECONDS and (
        classified_seconds < MIN_SUMMARY_ANALYZED_SECONDS
        or analyzed_fraction < MIN_INSUFFICIENT_COVERAGE_FRACTION
    ):
        state = "INSUFFICIENT_COVERAGE"
    elif duration < MIN_SUMMARY_ANALYZED_SECONDS:
        state = (
            "PARTIAL_COVERAGE"
            if analyzed_fraction >= MIN_INSUFFICIENT_COVERAGE_FRACTION
            else "INSUFFICIENT_COVERAGE"
        )
    elif (
        classified_seconds >= MIN_SUMMARY_ANALYZED_SECONDS
        and analyzed_fraction >= MIN_ADEQUATE_COVERAGE_FRACTION
    ):
        state = "ADEQUATE_PROVISIONAL_COVERAGE"
    else:
        state = "PARTIAL_COVERAGE"

    return {
        "recording_duration_seconds": duration,
        "total_windows": len(windows),
        "successfully_classified_windows": len(classified),
        "flagged_classified_windows": len(flagged),
        "unflagged_classified_windows": len(unflagged),
        "skipped_windows": len(skipped),
        "failed_windows": len(failed),
        "unprocessed_windows": len(unprocessed),
        "classified_all_seconds": classified_seconds,
        "classified_unflagged_seconds": unflagged_seconds,
        "skipped_window_interval_union_seconds": skipped_seconds,
        "failed_window_interval_union_seconds": failed_seconds,
        "unprocessed_seconds": max(0.0, duration - completed_seconds),
        "unanalyzed_seconds": max(0.0, duration - classified_seconds),
        "analyzed_time_fraction": analyzed_fraction,
        "unflagged_time_fraction": min(1.0, unflagged_seconds / duration),
        "coverage_state": state,
        "coverage_thresholds_are_provisional": True,
    }


def class_statistics(
    windows: Sequence[Mapping], labels: Sequence[str], coverage_state: str
) -> dict | None:
    """Return overlapping-window summaries only with adequate coverage."""
    if coverage_state != "ADEQUATE_PROVISIONAL_COVERAGE":
        return None

    successful = [window for window in windows if window.get("status") == "classified"]
    groups = {
        "all_classified_windows": successful,
        "unflagged_classified_windows": [
            window for window in successful if not window.get("quality_flags")
        ],
    }
    output: dict[str, dict] = {}
    for group_name, items in groups.items():
        per_class: dict[str, dict | None] = {}
        for label in labels:
            values = [float(window["scores_by_label"][label]) for window in items]
            per_class[label] = (
                {
                    "mean": mean(values),
                    "median": median(values),
                    "maximum": max(values),
                    "window_count": len(values),
                }
                if values
                else None
            )
        output[group_name] = per_class
    return output


def rank_candidates(
    windows: Sequence[Mapping],
    selected_label: str,
    verified_labels: Sequence[str],
) -> list[dict]:
    """Rank successful windows deterministically by a verified raw class.

    ``max(angry, fearful)`` is an optional selector only when both exact raw
    labels exist. Ties are ordered by earlier start time, then window index.
    """
    combined = selected_label == "max(angry, fearful)"
    if combined and not {"angry", "fearful"}.issubset(set(verified_labels)):
        raise ValueError("Combined selector requires raw labels angry and fearful")
    if not combined and selected_label not in verified_labels:
        raise ValueError(f"Unknown raw model label: {selected_label}")

    candidates = []
    for window in windows:
        if window.get("status") != "classified":
            continue
        scores = window["scores_by_label"]
        if combined:
            chosen_label = max(
                ("angry", "fearful"), key=lambda label: scores[label]
            )
            score = float(scores[chosen_label])
        else:
            chosen_label = selected_label
            score = float(scores[selected_label])
        candidate = dict(window)
        candidate["selected_class"] = chosen_label
        candidate["selected_score"] = score
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -item["selected_score"],
            float(item["start_seconds"]),
            int(item["index"]),
        )
    )
    return candidates
