"""Small, Streamlit-independent helpers for upload-bound session data."""

from __future__ import annotations

from collections.abc import MutableMapping

UPLOAD_HASH_KEY = "voxtrace_upload_sha256"
UPLOAD_FILENAME_KEY = "voxtrace_upload_filename"
ANALYSIS_SESSION_KEYS = (
    "voxtrace_results",
    "voxtrace_coverage",
    "voxtrace_class_statistics",
    "voxtrace_receipt",
    "voxtrace_audio_data",
    "voxtrace_loaded_hash",
    "voxtrace_audio_error",
    "voxtrace_analysis_error",
    "voxtrace_fingerprint_result",
    "voxtrace_fingerprint_input",
    "voxtrace_window_filter",
    "voxtrace_window_order",
    "voxtrace_window_selected",
    "voxtrace_result_tabs",
)


def clear_upload_state(state: MutableMapping) -> None:
    """Clear all analysis and fingerprint state when the upload is removed."""
    for key in (*ANALYSIS_SESSION_KEYS, UPLOAD_HASH_KEY, UPLOAD_FILENAME_KEY):
        state.pop(key, None)


def bind_state_to_upload(
    state: MutableMapping,
    current_sha256: str,
    current_filename: str | None = None,
) -> bool:
    """Clear prior upload state when its bytes or displayed filename change.

    Returns true when this upload identity is new or changed. The receipt
    records both the original-byte fingerprint and filename, so a rename must
    also clear previous analysis before a matching receipt is shown.
    """
    previous = (state.get(UPLOAD_HASH_KEY), state.get(UPLOAD_FILENAME_KEY))
    current = (current_sha256, current_filename)
    changed = previous != current
    if changed:
        for key in ANALYSIS_SESSION_KEYS:
            state.pop(key, None)
        state[UPLOAD_HASH_KEY] = current_sha256
        state[UPLOAD_FILENAME_KEY] = current_filename
    return changed
