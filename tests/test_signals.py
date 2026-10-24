from __future__ import annotations

import pytest

from config import EXPECTED_ID2LABEL
from signals import class_statistics, coverage_accounting, interval_union_seconds, merge_intervals, rank_candidates


LABELS = tuple(EXPECTED_ID2LABEL[index] for index in sorted(EXPECTED_ID2LABEL))


def result(index: int, start: float, end: float, status="classified", flags=None):
    return {
        "index": index,
        "start_seconds": start,
        "end_seconds": end,
        "status": status,
        "quality_flags": list(flags or []),
        "scores_by_label": {label: 1.0 / len(LABELS) for label in LABELS},
    }


def test_interval_union_overlapping_adjacent_and_disjoint():
    assert merge_intervals([(2, 3), (0, 1), (0.5, 2), (5, 7)]) == [(0.0, 3.0), (5.0, 7.0)]
    assert interval_union_seconds([(0, 3), (1.5, 4.5), (4.5, 5)]) == 5.0


@pytest.mark.parametrize("interval", [(1, 1), (2, 1), (-1, 1), (0, float("inf"))])
def test_interval_union_rejects_invalid_bounds(interval):
    with pytest.raises(ValueError):
        merge_intervals([interval])


def test_thirty_windows_with_only_two_successes_use_unique_coverage():
    windows = [
        result(index, index * 1.5, min(index * 1.5 + 3, 46.5),
               status="classified" if index < 2 else ("skipped" if index % 2 else "failed"))
        for index in range(30)
    ]
    summary = coverage_accounting(46.5, windows)
    assert summary["total_windows"] == 30
    assert summary["successfully_classified_windows"] == 2
    assert summary["skipped_windows"] + summary["failed_windows"] == 28
    assert summary["classified_all_seconds"] == 4.5
    assert summary["unanalyzed_seconds"] == 42.0
    assert summary["skipped_window_interval_union_seconds"] > 0
    assert summary["failed_window_interval_union_seconds"] > 0
    assert summary["unprocessed_seconds"] == 0.0
    assert summary["coverage_state"] == "INSUFFICIENT_COVERAGE"


def test_failed_overlap_does_not_remove_successfully_classified_time():
    windows = [
        result(0, 0, 3, "classified"),
        result(1, 1.5, 4.5, "failed"),
    ]
    summary = coverage_accounting(4.5, windows)
    assert summary["classified_all_seconds"] == 3.0
    assert summary["unanalyzed_seconds"] == 1.5
    assert summary["failed_windows"] == 1
    assert summary["failed_window_interval_union_seconds"] == 3.0
    assert summary["unprocessed_seconds"] == 0.0


def test_unprocessed_time_is_union_complement_of_completed_window_outcomes():
    windows = [
        result(0, 0, 2, "classified"),
        result(1, 3, 4, "pending"),
    ]
    summary = coverage_accounting(5.0, windows)
    assert summary["classified_all_seconds"] == 2.0
    assert summary["unprocessed_seconds"] == 3.0
    assert summary["unprocessed_windows"] == 1


def test_fully_classified_three_to_five_second_clip_is_partial_not_empty():
    summary = coverage_accounting(4.0, [result(0, 0, 3), result(1, 1.5, 4)])
    assert summary["classified_all_seconds"] == 4.0
    assert summary["coverage_state"] == "PARTIAL_COVERAGE"


@pytest.mark.parametrize(
    ("duration", "windows", "expected"),
    [
        (10.0, [], "NO_ANALYZED_AUDIO"),
        (20.0, [result(0, 0, 2)], "INSUFFICIENT_COVERAGE"),
        (12.0, [result(0, 0, 5.99)], "INSUFFICIENT_COVERAGE"),
        (12.0, [result(0, 0, 6)], "ADEQUATE_PROVISIONAL_COVERAGE"),
        (20.0, [result(0, 0, 8)], "PARTIAL_COVERAGE"),
        (5.0, [result(0, 0, 5)], "PARTIAL_COVERAGE"),
    ],
)
def test_coverage_state_boundaries(duration, windows, expected):
    assert coverage_accounting(duration, windows)["coverage_state"] == expected


def test_unflagged_union_excludes_flagged_windows_but_flagged_remain_counted():
    windows = [
        result(0, 0, 3, flags=[]),
        result(1, 1.5, 4.5, flags=["LOW_RMS_ENERGY"]),
    ]
    summary = coverage_accounting(4.5, windows)
    assert summary["classified_all_seconds"] == 4.5
    assert summary["classified_unflagged_seconds"] == 3.0
    assert summary["flagged_classified_windows"] == 1
    assert summary["unflagged_classified_windows"] == 1


def test_file_statistics_are_suppressed_without_adequate_coverage():
    windows = [result(0, 0, 3)]
    assert class_statistics(windows, LABELS, "PARTIAL_COVERAGE") is None
    assert class_statistics(windows, LABELS, "INSUFFICIENT_COVERAGE") is None


def test_class_statistics_include_all_and_unflagged_groups():
    windows = [
        result(0, 0, 3, flags=[]),
        result(1, 1.5, 4.5, flags=["TONAL_SUSPECTED"]),
    ]
    stats = class_statistics(windows, LABELS, "ADEQUATE_PROVISIONAL_COVERAGE")
    assert stats["all_classified_windows"]["angry"]["window_count"] == 2
    assert stats["unflagged_classified_windows"]["angry"]["window_count"] == 1


def test_candidate_ranking_is_stable_and_flagged_windows_remain_inspectable():
    windows = [
        result(3, 3.0, 6.0, flags=[]),
        result(2, 1.5, 4.5, flags=["LOW_VAD_ACTIVITY"]),
        result(1, 1.5, 4.5, flags=[]),
    ]
    windows[0]["scores_by_label"]["angry"] = 0.7
    windows[1]["scores_by_label"]["fearful"] = 0.7
    windows[2]["scores_by_label"]["angry"] = 0.7
    unflagged = [item for item in windows if not item["quality_flags"]]
    flagged = [item for item in windows if item["quality_flags"]]
    assert [item["index"] for item in rank_candidates(unflagged, "angry", LABELS)] == [1, 3]
    assert [item["index"] for item in rank_candidates(flagged, "max(angry, fearful)", LABELS)] == [2]
    assert rank_candidates(flagged, "max(angry, fearful)", LABELS)[0]["selected_class"] == "fearful"


def test_combined_ranking_selector_requires_exact_raw_labels():
    with pytest.raises(ValueError):
        rank_candidates([], "max(angry, fearful)", ("angry", "fear"))
