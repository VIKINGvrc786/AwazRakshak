# AwazRakshak

AwazRakshak is a local, single-purpose Streamlit app: start browser-microphone capture, view uncalibrated voice-model scores as they arrive, then stop and save an audio-free call summary. Audio remains in a rolling in-memory buffer and is discarded when recording stops.

## Setup

Use Python 3.12 on Windows and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In the app, select the local model folder, confirm that you are authorised to analyse the call, and click **Start recording**. Allow microphone access in the browser. Click **Stop and save summary** when finished. The JSON summary is written to `.awazrakshak-data/summaries/` and can also be downloaded from the page. The summary contains no audio.

## Pinned local model

The app never downloads model files. Enter either the Hugging Face cache root, the pinned revision's snapshot folder, or a folder containing `config.json`, `preprocessor_config.json`, and `model.safetensors`. AwazRakshak verifies the pinned model metadata and all three file hashes before enabling recording. A missing file, changed hash, incorrect label map, or mismatched revision disables analysis and displays the reason.

## Limitations

Scores and derived indicators are uncalibrated and provisional. They support human review only and do not determine danger, deception, guilt, or dispatch.
