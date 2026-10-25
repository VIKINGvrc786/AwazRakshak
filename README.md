# AwazRakshak

AwazRakshak (formerly VoxTrace) is an experimental, local investigator-support prototype for reviewing selected portions of recorded WAV audio alongside raw outputs from a pinned Wav2Vec2 emotion-classification model.

> **Important:** AwazRakshak is a review and screening aid, not a lie detector, emergency detector, clinical tool, or automated decision system. Model scores are uncalibrated outputs and must not be interpreted as proof that a person is experiencing an emotion or that an audio recording indicates danger, deception, guilt, fraud, malicious intent, or urgency.

## What AwazRakshak does

- Loads and reviews WAV recordings locally.
- Splits usable audio into overlapping analysis windows.
- Converts supported multichannel audio to mono and resamples it to the verified 16 kHz model rate when required.
- Runs a pinned Wav2Vec2 emotion-classification checkpoint locally after the required model files are available.
- Displays the model's raw seven-class output scores.
- Provides deterministic window ranking and listen-first review queues.
- Reports acoustic-quality heuristics such as low RMS, low VAD activity, clipping, and tonal-suspect signals.
- Calculates processing coverage without double-counting overlapping windows.
- Generates a versioned JSON evidence receipt containing metadata, transformations, model identity, scores, flags, coverage, thresholds, and dependency information.
- Keeps uploaded audio out of Git by default.

## What AwazRakshak does NOT do

The pinned checkpoint exposes these raw classes:

| ID | Raw label |
|---:|---|
| 0 | `angry` |
| 1 | `calm` |
| 2 | `disgust` |
| 3 | `fearful` |
| 4 | `happy` |
| 5 | `sad` |
| 6 | `surprised` |

AwazRakshak does not independently detect `nervousness` or `distress`, and it does not synthesize either label. In particular, `calm` is not renamed to `neutral`.

The classifier can produce scores for noise, music, silence-like inputs, tones, and unrelated audio. A high score on non-speech audio does not establish that an emotion is present. Low or missing scores do not establish safety or calm.

AwazRakshak must not be used to trigger emergency dispatch, operational triage, disciplinary action, medical decisions, or other high-impact decisions without independent human review and appropriate validation.

## Architecture at a glance

```text
WAV file
   |
   v
Decode + validate
   |
   +--> metadata / SHA-256
   |
   v
Mono conversion + 16 kHz resampling
   |
   v
3 s windows / 1.5 s stride
   |
   +--> acoustic/VAD quality checks
   |
   v
Pinned Wav2Vec2 checkpoint
   |
   v
7 raw class scores
   |
   +--> deterministic ranking
   +--> listen-first review queue
   +--> coverage calculation
   |
   v
Versioned JSON receipt
```

## Requirements

The project is tested with:

- Windows
- Python 3.12 (the current tested environment used Python 3.12.10)
- The pinned dependency versions in [`requirements.txt`](requirements.txt)

The main dependencies include Streamlit, NumPy, pandas, SoundFile, soxr, PyTorch, Transformers, Hugging Face Hub, safetensors, and pytest. `webrtcvad-wheels` is optional for quality-flagging support; usable audio can still be classified if VAD is unavailable.

## Quick start — Windows

Open PowerShell in the repository directory:

```powershell
cd C:\path\to\VoxTrace

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify the exact model pin before starting the application:

```powershell
.\.venv\Scripts\python.exe scripts\verify_model.py
```

Start AwazRakshak:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The application binds to `127.0.0.1` and normally opens at:

```text
http://127.0.0.1:8501
```

### First-run model download

The first verification/run may require access to Hugging Face if the pinned model is not already cached. Inference is performed locally; the application does not send uploaded recordings to an external inference service.

After the model and dependencies have been cached, test the offline path with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_model.py --offline
```

Do not describe the application as fully offline until this check succeeds on the current checkout.

## Model verification

The production checkpoint is pinned to:

```text
Dpngtm/wav2vec2-emotion-recognition
revision: 033a5751a5bbe5b0b67c2c71e6102c38de35a346
```

The verifier checks the pinned revision/local Hub cache, loads the configuration and feature extractor, validates the safetensors weights, checks the expected class metadata and output dimensions, and performs a synthetic forward pass.

If the verifier detects a missing or mismatched model pin, label mapping, or output dimension, analysis is disabled rather than silently accepting an unverified checkpoint.

The current repository's model-verification record is stored in [`MODEL_VERIFICATION.json`](MODEL_VERIFICATION.json). It records `human_approved: false`; it does not claim human approval. The model card identifies TESS, CREMA-D, SAVEE, and RAVDESS as training sources, but this project has not independently validated the model's real-world emotion accuracy or performed a leakage analysis.

## Analysis pipeline

### Input validation

- WAV metadata is recorded, including original format/subtype, sample rate, channels, frame count, and duration.
- Non-finite samples reject the recording before conversion.
- Empty/corrupt files and recordings shorter than one second are rejected.
- Files larger than 50 MB are rejected.

### Audio transformation

- Multichannel audio is averaged to mono.
- Audio is resampled to the verified 16 kHz model rate with `soxr` when necessary.

### Windowing

- Window length: 3 seconds.
- Stride: 1.5 seconds.
- The final partial window is retained when it is at least 1 second long.
- Only unusable short windows and effectively digital-silent windows are skipped.

Quiet, tonal, stationary-noise-like, low-VAD, and clipped audio may still be classified and receives non-blocking quality flags.

### Acoustic quality flags

`TONAL_SUSPECTED` is raised using spectral-concentration heuristics. It does not prove that speech is absent; harmonic or voiced audio can also trigger the heuristic.

If VAD is unavailable, usable audio is still classified and receives `VAD_UNAVAILABLE`.

### Coverage

Processing coverage is calculated from the union of successful time intervals so overlapping windows do not inflate analyzed duration.

Short clips below six seconds are visibly treated as partial even when all usable audio is processed. File-level class summaries are only shown when provisional coverage is adequate.

## Review workspace

The Streamlit interface separates the workflow into overview, window review, and evidence areas.

The window-review workspace provides:

1. A timestamped window list.
2. Deterministic selection and sorting.
3. An unflagged listen-first queue.
4. A separate flagged queue that remains available for playback.
5. The selected interval's audio player.
6. The literal top model output and raw score.
7. The runner-up class and ties at four-decimal display precision.
8. All seven class scores.
9. Expandable acoustic diagnostics.

The default ranking expression is `max(angry, fearful)` because both exact labels are present in the verified mapping. Ties are resolved by earlier start time and then lower window index; an equal angry/fearful score selects `angry` first.

## Provisional screening values

These are engineering starting points, not scientifically validated thresholds:

| Setting | Initial value |
|---|---:|
| Window / stride | 3.0 s / 1.5 s |
| Minimum analyzable window | 1.0 s |
| Digital-silence RMS floor | 0.0001 |
| Low-RMS flag | 0.005 |
| Low VAD activity flag | 0.20 |
| Minimum analyzed duration for adequate coverage | 6 s |
| Coverage distinction | 25% / 50% |

The values are defined in `config.py` and included in each receipt. Quality flags are heuristics, not proof that speech is absent.

## Evidence receipts

AwazRakshak produces a strict JSON receipt, currently version 3.1.

The receipt records:

- Uploaded filename.
- SHA-256 hash of the original uploaded bytes.
- WAV metadata.
- Audio transformations.
- Pinned model identity.
- Raw class scores.
- Per-window status and flags.
- Coverage information.
- Thresholds.
- Dependency information.

The receipt does not contain audio bytes or full audio arrays. Version 3.1 adds the optional `input.filename` field while retaining compatibility with version 3.0 receipts.

A SHA-256 match proves byte equality only. Review filenames before sharing receipts because filenames can contain identifying information.

## Testing

Run the standard test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

The normal unit/synthetic tests do not require downloading the model.

For the real-model integration test, use a locally cached model:

```powershell
$env:VOXTRACE_RUN_REAL_MODEL_TESTS = "1"
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -m pytest -v -m real_model
```

Passing synthetic tests does not establish real-world emotion-classification accuracy.

## Five-minute demo

1. **Scope:** Explain that AwazRakshak does not determine danger, deception, guilt, distress, urgency, or dispatch decisions.
2. **Verification:** Show the immutable checkpoint revision, verification result, seven raw labels, and 16 kHz requirement.
3. **Consented recording:** Use a short team-recorded WAV for which all speakers have consented. State that the example is anecdotal and is not a benchmark.
4. **Review:** Inspect metadata, coverage, raw scores, quality flags, and playback.
5. **Synthetic audio:** Generate a tone or noise fixture to demonstrate that non-speech audio can still receive model scores.
6. **Receipt:** Export the JSON receipt and inspect the source-byte hash, transformations, window results, coverage, and thresholds.
7. **Tests:** Show the actual `pytest -v` output and only claim offline-after-cache behavior if the offline verification command succeeds on the current checkout.

Generate a simple tone fixture with:

```powershell
.\.venv\Scripts\python.exe -c "import numpy as np, soundfile as sf; sr=16000; t=np.arange(sr*5)/sr; sf.write('tone_demo.wav', .1*np.sin(2*np.pi*440*t), sr)"
```

Keep demonstration recordings consented and local. WAV files should not be committed to the repository.

## Troubleshooting

**`py` cannot find Python 3.12**

Install Python 3.12 and confirm with:

```powershell
py -3.12 --version
```

**Model verification fails**

Do not modify expected labels or dimensions to force verification to pass. Reinstall the pinned dependencies and confirm that the exact model revision is available.

**Offline verification fails**

Run the normal verifier once with network access so the exact pinned model files can be cached, then rerun:

```powershell
.\.venv\Scripts\python.exe scripts\verify_model.py --offline
```

**VAD is unavailable**

This is non-fatal. AwazRakshak can continue classifying usable audio and records the `VAD_UNAVAILABLE` condition.

**A recording receives a surprising emotion score**

Listen to the selected window and inspect the acoustic flags. The checkpoint can score non-speech audio, and its raw outputs are not calibrated probabilities.

## Repository structure

```text
VoxTrace/
├── app.py
├── config.py
├── requirements.txt
├── MODEL_VERIFICATION.json
├── LICENSE
├── LICENSES.md
├── README.md
├── scripts/
│   └── verify_model.py
└── tests/
    └── ...
```

> **Repository name:** The GitHub repository is currently still named `VoxTrace`. The application/project name is now **AwazRakshak**. Rename the GitHub repository separately if you want the URL/slug to match.

## Licensing

The AwazRakshak source code is released under the MIT License. See [`LICENSE`](LICENSE).

Third-party model, base-model, and training-dataset terms are tracked separately in [`LICENSES.md`](LICENSES.md). The checkpoint repository currently states MIT metadata, while the underlying training datasets have different terms, including noncommercial and organization-specific restrictions. Therefore, the MIT license for this source repository should not be interpreted as granting unrestricted rights to the third-party checkpoint, training data, or demo recordings.

Before public redistribution of model weights or commercial deployment, review the checkpoint provenance and the applicable upstream dataset/model terms. Do not redistribute source datasets or recorded audio without the required permissions.

## Privacy and data handling

AwazRakshak is designed for local review. Uploaded recordings are processed by the local application/model rather than being sent to a cloud inference endpoint. The application records a SHA-256 fingerprint and metadata in its evidence receipt; a receipt may therefore contain information that can identify or correlate with an input file.

Do not upload recordings containing sensitive personal information unless you have an appropriate legal basis and consent to process them.

## Status

AwazRakshak is an experimental research/prototype project. Its model outputs and screening heuristics should be treated as investigative aids requiring human review, not as validated evidence or production-grade emotion measurement.

## Contributing

Contributions are welcome. For changes that affect model verification, label mappings, thresholds, receipt schemas, or evidence semantics, include tests and document the behavioral change clearly.

Please avoid committing:

- Audio recordings.
- Model weights or caches.
- Credentials, API tokens, or private data.
- Generated receipts containing identifying information.

## License and third-party notices

- **AwazRakshak source:** MIT License — see [`LICENSE`](LICENSE).
- **Third-party model/dataset review:** see [`LICENSES.md`](LICENSES.md).
- **Model provenance and licensing status:** see [`LICENSES.md`](LICENSES.md) before redistribution.
