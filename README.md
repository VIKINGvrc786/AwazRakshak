# VoxTrace

VoxTrace is an experimental, local investigator-support prototype for listening to selected portions of recorded WAV audio alongside raw outputs from a pinned Wav2Vec2 emotion classifier.

## Scope and limits

The classifier emits a closed set of labels. The pinned revision currently verifies these raw classes: `angry`, `calm`, `disgust`, `fearful`, `happy`, `sad`, and `surprised`. VoxTrace directly displays the checkpoint's `angry` and `fearful` scores; it does not independently detect `nervousness` or `distress`, and does not synthesize either label. The app does not rename labels. In particular, `calm` is not `neutral`.

Nervousness and distress are not independent classes in this checkpoint. Scores are uncalibrated classifier outputs, not probabilities that an emotion truly exists, that the caller experiences it, or that the model is correct. The classifier also emits scores for noise, music, silence-like inputs, and unrelated audio. A large score on a tone or noise does not establish an emotion.

VoxTrace does not determine danger, emergency severity, risk to life, urgency, deception, truthfulness, guilt, fraud, malicious intent, or clinical status. It must not trigger dispatch or operational triage. Low or missing scores do not establish safety or calm. Human listening and independent investigation remain necessary.

The model card names TESS, CREMA-D, SAVEE, and RAVDESS as training sources. No independent performance evaluation or leakage analysis was conducted for this project. Do not present model-card metrics as independent validation.

## Windows setup

Use Python 3.12. The existing checked environment used Python 3.12.10 and the pinned package versions in `requirements.txt`.

```powershell
cd C:\Users\varun\Desktop\voxtrace
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\verify_model.py
.\.venv\Scripts\streamlit.exe run app.py
```

Streamlit binds to `127.0.0.1` and opens at `http://127.0.0.1:8501`. Hugging Face access is needed the first time the pinned model is not cached; the app does not use cloud inference or send recordings to external services. Hugging Face telemetry and Streamlit usage-stat collection are disabled. Do not describe the app as offline until the local model and dependencies are cached; run `python scripts\verify_model.py --offline` to check the pinned model cache.

If the verifier reports a missing or mismatched pin, label map, or output dimension, analysis remains disabled. Do not change the expected labels to make an unverified checkpoint pass.

## Analysis behavior

- WAV decoding records original format, subtype, sampling rate, channels, frames, and duration. Non-finite samples reject the complete recording before conversion. Recordings shorter than one second, empty or corrupt files, and files over 50 MB are rejected.
- Supported multichannel audio is averaged to mono. Audio is resampled to the verified 16 kHz model rate with `soxr` when necessary.
- Windows are up to 3 seconds with a 1.5-second stride. The final partial window is retained when it is at least one second long.
- Only unusable short windows and effectively digital-silent windows are skipped. Quiet, tonal, stationary-noise-like, low-VAD, and clipped audio can still be classified and receives non-blocking flags. If VAD is unavailable, usable audio is still classified and receives `VAD_UNAVAILABLE`.
- Processing coverage uses the union of successful time intervals, so overlapping windows do not inflate analyzed time. Short clips below six seconds are visibly partial even if fully processed. File-level class summaries appear only with adequate provisional coverage.
- The unflagged listen-first queue is shown first. The flagged queue remains available for playback. Ranking uses a selected raw class; the default is `max(angry, fearful)` only because both exact labels are present in the verified mapping. Ties use earlier start time then lower window index; an equal angry/fearful score selects `angry` first.
- The JSON receipt is strict JSON version 3.0. It records the SHA-256 of the original uploaded bytes, metadata, transformations, pinned model identity, output scores, window statuses and flags, coverage, thresholds, and dependencies. It excludes audio bytes and arrays. Matching SHA-256 values prove byte equality only.

## Provisional screening and coverage values

These are engineering starting points, not scientific thresholds:

| Setting | Initial value |
| --- | ---: |
| Window / stride | 3.0 s / 1.5 s |
| Minimum analyzable window | 1.0 s |
| Digital-silence RMS floor | 0.0001 |
| Low-RMS flag | 0.005 |
| Low VAD activity flag | 0.20 |
| Minimum analyzed duration for adequate coverage | 6 s |
| Minimum coverage fraction for insufficient/adequate distinction | 25% / 50% |

All values are named in `config.py` and are included in each receipt. Quality flags are heuristics; they are not proof that speech is absent.

## Model verification

The production pin is `Dpngtm/wav2vec2-emotion-recognition` at `033a5751a5bbe5b0b67c2c71e6102c38de35a346`. Verification inspects the remote revision when available and the local Hub cache, loads configuration, feature extractor, and safetensors weights at the exact SHA, checks class metadata and head modules, and performs a synthetic forward pass. The current environment’s record is in `MODEL_VERIFICATION.json`; it records `human_approved: false` and does not assert human approval.

The observed raw map is:

| ID | Raw label |
| ---: | --- |
| 0 | `angry` |
| 1 | `calm` |
| 2 | `disgust` |
| 3 | `fearful` |
| 4 | `happy` |
| 5 | `sad` |
| 6 | `surprised` |

The model card’s narrative description and the configuration’s raw labels differ. Runtime checks use the pinned configuration and fail closed if the observed mapping or seven-logit output changes.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Unit and synthetic tests do not download a model. The separate real-model test can be run with the model already cached:

```powershell
$env:VOXTRACE_RUN_REAL_MODEL_TESTS = "1"
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -m pytest -v -m real_model
```

The tests use synthetic WAV fixtures and do not establish real-world emotion accuracy.

## Five-minute demo script

1. **0:00–0:35 — Scope.** Read the scope banner and point out that the tool does not decide danger, deception, guilt, distress, urgency, or dispatch.
2. **0:35–1:05 — Verified model.** Show the full immutable revision, verification status, seven observed raw labels, and the model’s 16 kHz feature-extractor requirement.
3. **1:05–2:10 — Consented recording.** Upload a short team-recorded WAV with all speakers’ consent. State that the example is anecdotal, not a benchmark or evidence of accuracy. Review metadata, coverage, raw outputs, and playback.
4. **2:10–3:10 — Screening and queues.** Compare the unflagged and flagged queues. Open a flagged window and listen to it; explain that flags are heuristics and do not discard the window.
5. **3:10–4:00 — Synthetic non-speech.** Generate and upload a tone or noise fixture. Show that it can still receive closed-set scores and flags, demonstrating why a score alone is not evidence of a true emotion.

   ```powershell
   .\.venv\Scripts\python.exe -c "import numpy as np, soundfile as sf; sr=16000; t=np.arange(sr*5)/sr; sf.write('tone_demo.wav', .1*np.sin(2*np.pi*440*t), sr)"
   ```

6. **4:00–4:35 — Receipt.** Download the JSON receipt and point to the source-byte fingerprint, transformation, window results, flags, coverage, and provisional thresholds. Leave the verification box blank, then enter a deliberately changed fingerprint and show the mismatch.
7. **4:35–5:00 — Evidence.** Show the actual `pytest -v` results. Mention offline-after-cache behavior only if the offline integration command above has been run successfully on the current checkout.

Keep all demo recordings consented and local. WAV files are ignored by Git and should not be added to the repository.

## Licensing review

See [`LICENSES.md`](LICENSES.md) for separately recorded checkpoint, base-model, and training-dataset license information, source URLs, verification date, and unresolved rights questions. The checkpoint’s MIT metadata does not by itself settle derivative-weight rights. Confirm hackathon-use permissions with the organizers where needed.
