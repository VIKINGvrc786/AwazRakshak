"""Application, model, audio, and provisional screening configuration."""

from __future__ import annotations

MODEL_ID = "Dpngtm/wav2vec2-emotion-recognition"
MODEL_REVISION = "033a5751a5bbe5b0b67c2c71e6102c38de35a346"
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
WINDOW_STRIDE_SECONDS = 1.5
MIN_ANALYZABLE_WINDOW_SECONDS = 1.0
MAX_UPLOAD_MB = 50
LONG_RECORDING_SECONDS = 120

# Engineering starting points only. None of these values are validated
# scientific thresholds or emotion decision boundaries.
DIGITAL_SILENCE_RMS_FLOOR = 1e-4
LOW_RMS_FLAG_THRESHOLD = 0.005
CLIPPING_PEAK_THRESHOLD = 0.999
LOW_VAD_ACTIVITY_RATIO = 0.20
TONAL_SPECTRAL_FLATNESS_MAX = 0.035
TONAL_DOMINANT_POWER_FRACTION = 0.35
STATIONARY_NOISE_RMS_CV_MAX = 0.15
STATIONARY_NOISE_SPECTRAL_FLATNESS_MIN = 0.08

MIN_SUMMARY_ANALYZED_SECONDS = 6.0
MIN_INSUFFICIENT_COVERAGE_FRACTION = 0.25
MIN_ADEQUATE_COVERAGE_FRACTION = 0.50
SCORE_SUM_TOLERANCE = 1e-4

PROVISIONAL_THRESHOLDS = {
    "window_seconds": WINDOW_SECONDS,
    "window_stride_seconds": WINDOW_STRIDE_SECONDS,
    "minimum_analyzable_window_seconds": MIN_ANALYZABLE_WINDOW_SECONDS,
    "digital_silence_rms_floor": DIGITAL_SILENCE_RMS_FLOOR,
    "low_rms_flag_threshold": LOW_RMS_FLAG_THRESHOLD,
    "clipping_peak_threshold": CLIPPING_PEAK_THRESHOLD,
    "low_vad_activity_ratio": LOW_VAD_ACTIVITY_RATIO,
    "tonal_spectral_flatness_max": TONAL_SPECTRAL_FLATNESS_MAX,
    "tonal_dominant_power_fraction": TONAL_DOMINANT_POWER_FRACTION,
    "stationary_noise_rms_cv_max": STATIONARY_NOISE_RMS_CV_MAX,
    "stationary_noise_spectral_flatness_min": STATIONARY_NOISE_SPECTRAL_FLATNESS_MIN,
    "minimum_summary_analyzed_seconds": MIN_SUMMARY_ANALYZED_SECONDS,
    "minimum_insufficient_coverage_fraction": MIN_INSUFFICIENT_COVERAGE_FRACTION,
    "minimum_adequate_coverage_fraction": MIN_ADEQUATE_COVERAGE_FRACTION,
    "score_sum_tolerance": SCORE_SUM_TOLERANCE,
    "status": "provisional engineering starting points; not scientifically validated",
}
