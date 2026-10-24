"""Claim-safe copy shared by the interface and static-copy tests."""

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

COVERAGE_DESCRIPTION = (
    "Coverage describes only which time intervals were processed. It does not "
    "validate model outputs or establish safety. Overlapping windows are counted "
    "once in the time totals."
)
