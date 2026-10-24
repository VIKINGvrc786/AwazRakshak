"""Small, Streamlit-independent helpers for upload-bound session data."""

from __future__ import annotations

from collections.abc import MutableMapping

UPLOAD_HASH_KEY = "voxtrace_upload_sha256"
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
)


def clear_upload_state(state: MutableMapping) -> None:
    """Clear all analysis and fingerprint state when the upload is removed."""
    for key in (*ANALYSIS_SESSION_KEYS, UPLOAD_HASH_KEY):
        state.pop(key, None)


def bind_state_to_upload(state: MutableMapping, current_sha256: str) -> bool:
    """Clear prior upload state on any byte change and bind to the new hash.

    Returns true when this is a new or changed upload. Filename changes with
    identical bytes do not invalidate results because the receipt fingerprints
    the original bytes.
    """
    previous = state.get(UPLOAD_HASH_KEY)
    changed = previous != current_sha256
    if changed:
        for key in ANALYSIS_SESSION_KEYS:
            state.pop(key, None)
        state[UPLOAD_HASH_KEY] = current_sha256
    return changed
