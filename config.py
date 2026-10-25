"""Pinned local model and live screening configuration."""

from __future__ import annotations

MODEL_ID = "Dpngtm/wav2vec2-emotion-recognition"
MODEL_REVISION = "033a5751a5bbe5b0b67c2c71e6102c38de35a346"
MODEL_LOCAL_PATH_ENV = "AWAZRAKSHAK_MODEL_PATH"

EXPECTED_ID2LABEL = {
    0: "angry",
    1: "calm",
    2: "disgust",
    3: "fearful",
    4: "happy",
    5: "sad",
    6: "surprised",
}
EXPECTED_LABEL2ID = {label: index for index, label in EXPECTED_ID2LABEL.items()}
MODEL_OUTPUT_DIMENSION = len(EXPECTED_ID2LABEL)
MODEL_SAMPLE_RATE = 16_000
WINDOW_SECONDS = 3.0
LIVE_BUFFER_SECONDS = 30.0
LIVE_UPDATE_INTERVAL_SECONDS = 1.5
LIVE_SMOOTHING_ALPHA = 0.35
LIVE_CATEGORY_CONFIRM_WINDOWS = 2
LIVE_MAX_INSIGHTS = 2_000
LIVE_CATEGORY_THRESHOLDS = {"elevated": 0.35, "high": 0.65}

DIGITAL_SILENCE_RMS_FLOOR = 1e-4
LOW_RMS_FLAG_THRESHOLD = 0.005
CLIPPING_PEAK_THRESHOLD = 0.999
LOW_VAD_ACTIVITY_RATIO = 0.20
TONAL_SPECTRAL_FLATNESS_MAX = 0.035
TONAL_DOMINANT_POWER_FRACTION = 0.35
STATIONARY_NOISE_RMS_CV_MAX = 0.15
STATIONARY_NOISE_SPECTRAL_FLATNESS_MIN = 0.08

PROVISIONAL_INDICATOR_WEIGHTS = {
    "distress": {"fearful": 0.50, "sad": 0.30, "angry": 0.20},
    "nervousness": {"fearful": 0.45, "surprised": 0.35, "rms_cv": 0.20},
}
PROVISIONAL_ATTENTION_THRESHOLDS = {"elevated": 0.35, "high": 0.65}
LIVE_CATEGORY_THRESHOLDS = PROVISIONAL_ATTENTION_THRESHOLDS

INDICATOR_FORMULAS = {
    "distress_indicator": "0.50 * fearful + 0.30 * sad + 0.20 * angry",
    "nervousness_indicator": "0.45 * fearful + 0.35 * surprised + 0.20 * min(rms_cv, 1)",
    "attention_level": "Routine < 0.35; Elevated >= 0.35 and < 0.65; High >= 0.65",
}
SCORE_SUM_TOLERANCE = 1e-4

PROVISIONAL_THRESHOLDS = {
    "window_seconds": WINDOW_SECONDS,
    "digital_silence_rms_floor": DIGITAL_SILENCE_RMS_FLOOR,
    "low_rms_flag_threshold": LOW_RMS_FLAG_THRESHOLD,
    "clipping_peak_threshold": CLIPPING_PEAK_THRESHOLD,
    "low_vad_activity_ratio": LOW_VAD_ACTIVITY_RATIO,
    "tonal_spectral_flatness_max": TONAL_SPECTRAL_FLATNESS_MAX,
    "tonal_dominant_power_fraction": TONAL_DOMINANT_POWER_FRACTION,
    "stationary_noise_rms_cv_max": STATIONARY_NOISE_RMS_CV_MAX,
    "stationary_noise_spectral_flatness_min": STATIONARY_NOISE_SPECTRAL_FLATNESS_MIN,
    "indicator_weights": PROVISIONAL_INDICATOR_WEIGHTS,
    "attention_thresholds": PROVISIONAL_ATTENTION_THRESHOLDS,
    "smoothing_alpha": LIVE_SMOOTHING_ALPHA,
    "category_confirm_windows": LIVE_CATEGORY_CONFIRM_WINDOWS,
    "status": "Provisional engineering starting points; not scientifically validated.",
}
