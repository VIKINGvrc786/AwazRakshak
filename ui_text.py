"""Claim-safe interface copy and small display helpers."""

from config import TONAL_DOMINANT_POWER_FRACTION, TONAL_SPECTRAL_FLATNESS_MAX


def format_raw_score(score: float) -> str:
    """Format an uncalibrated model score consistently for display."""
    return f"{float(score):.4f}"


def rank_raw_scores(scores: list[dict]) -> list[dict]:
    """Return score rows highest-first, preserving input order for exact ties."""
    return sorted(scores, key=lambda row: float(row["score"]), reverse=True)


def summarize_top_outputs(scores: list[dict]) -> tuple[dict, dict | None, tuple[str, ...]]:
    """Return top, runner-up and labels tied at the displayed precision."""
    ranked = rank_raw_scores(scores)
    if not ranked:
        raise ValueError("A classified window must contain at least one raw score")
    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    top_display = format_raw_score(top["score"])
    tied_labels = tuple(
        row["label"]
        for row in ranked
        if format_raw_score(row["score"]) == top_display
    )
    return top, runner_up, tied_labels

SCOPE_BANNER = (
    "Experimental investigator-support prototype. VoxTrace displays raw "
    "outputs from a closed-set speech-emotion classifier for human review. "
    "It does not determine danger, urgency, deception, guilt, fraud, intent, "
    "or mental-health status, and it must not be used for dispatch or operational triage."
)

LOW_SCORE_NOTICE = (
    "A low score means this model did not strongly output that class for the "
    "analyzed windows. It does not mean the call is safe, calm, or free of fear, "
    "anger, distress or nervousness. Unanalyzed audio has not been assessed."
)

MODEL_LIMITATIONS = (
    "Anger and fear correspond only to the direct checkpoint labels `angry` "
    "and `fearful`, if the verified mapping confirms them. Nervousness and "
    "distress are not modelled as independent classes by this checkpoint. "
    "`calm` is not relabelled `neutral`; a calm-class score is not proof of "
    "neutral or non-emotional speech. This model always selects among its "
    "closed set, including for unrelated or non-speech audio; a large score "
    "on noise does not establish a true emotion."
)

QUEUE_TITLE = (
    "Listen-First Candidate Windows — Sorted by Raw Model Output, Not an Emotion or Threat Verdict"
)

SCORE_DESCRIPTION = (
    "Raw classifier scores are not calibrated probabilities that a real emotion "
    "exists, that a caller is experiencing it, or that the model is correct."
)

TONAL_FLAG_DESCRIPTION = (
    "`TONAL_SUSPECTED` is raised when positive-frequency spectral flatness is "
    f"≤ {TONAL_SPECTRAL_FLATNESS_MAX:.3f} or the dominant non-DC FFT-bin power "
    f"share is ≥ {TONAL_DOMINANT_POWER_FRACTION:.0%}. This marks concentrated "
    "spectral energy; harmonic or voiced-like audio can also trigger it. It is "
    "not a speech-presence or speech-absence test, and it is not an emotion or "
    "threat verdict. These are provisional engineering cutoffs, not validated "
    "speech criteria."
)

COVERAGE_DESCRIPTION = (
    "Coverage describes only which time intervals were processed. It does not "
    "validate model outputs or establish safety. Overlapping windows are counted "
    "once in the time totals."
)


def format_measurement(value: float | None, format_spec: str) -> str:
    """Format an available acoustic measurement without inventing a zero."""
    return "unavailable" if value is None else format(value, format_spec)


def safe_analysis_error_message(error: Exception) -> str:
    """Give a useful next step without exposing internal exception details."""
    error_type = type(error).__name__
    if error_type == "ModelVerificationError":
        return (
            "The pinned model could not be loaded or reverified from the local cache. "
            "Check the cached revision, then run `python scripts\\verify_model.py`."
        )
    if error_type == "AudioValidationError":
        return "The WAV could not be analyzed. Recheck its format, sample rate, and duration."
    return (
        f"Local analysis failed safely ({error_type}). No results or receipt were retained. "
        "Check the WAV and local model cache; run `python scripts\\verify_model.py` if the issue persists."
    )
