import pytest

from signals import (
    attention_level,
    categorize_call,
    derive_indicators,
    smooth_indicator,
)


def test_derived_indicator_formulas_and_smoothing():
    scores = {"fearful": 0.8, "sad": 0.5, "angry": 0.25, "surprised": 0.4}
    indicators = derive_indicators(scores, {"rms_cv": 0.6})

    assert indicators["distress_indicator"] == pytest.approx(0.6)
    assert indicators["nervousness_indicator"] == pytest.approx(0.62)
    assert smooth_indicator([0.2, 0.6], 0.5) == pytest.approx(0.4)


def test_indicators_reject_missing_or_invalid_inputs():
    with pytest.raises(ValueError, match="Missing raw scores"):
        derive_indicators({"fearful": 0.5}, {"rms_cv": 0.2})

    scores = {"fearful": 0.5, "sad": 0.2, "angry": 0.1, "surprised": 0.2}
    with pytest.raises(ValueError, match="rms_cv"):
        derive_indicators(scores, {})

    with pytest.raises(ValueError, match="within"):
        attention_level(1.1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "Low"), (0.349, "Low"), (0.35, "Elevated"), (0.65, "High")],
)
def test_attention_level_thresholds(value, expected):
    assert attention_level(value) == expected


def test_attention_category_requires_two_consecutive_windows_to_change():
    indicators = {"distress_indicator": 0.8, "nervousness_indicator": 0.1}

    assert categorize_call(
        indicators,
        [],
        previous_category="Routine",
        candidate_history=[],
    ) == "Routine"
    assert categorize_call(
        indicators,
        [],
        previous_category="Routine",
        candidate_history=["High"],
    ) == "High"
