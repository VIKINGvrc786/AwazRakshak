# AwazRakshak

**Local, real-time voice-signal review with human oversight.**

AwazRakshak is an experimental Python and Streamlit application that captures microphone audio in the browser, analyzes short audio windows with a locally stored, pinned speech-emotion model, displays indicative voice signals, and saves an audio-free JSON call summary.

It is designed to support a human reviewer—not to make decisions about a person or call.

> **Important limitation:** Model scores and derived distress/nervousness indicators are provisional and uncalibrated. They are not proof of a person's emotional state, intent, honesty, danger, guilt, or need for emergency response. Do not use this prototype as an automated dispatch, triage, surveillance, employment, legal, medical, or other high-stakes decision system.

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configure the local model](#configure-the-local-model)
- [Run the application](#run-the-application)
- [Use the app](#use-the-app)
- [Privacy and saved data](#privacy-and-saved-data)
- [Model details and limitations](#model-details-and-limitations)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [License](#license)

## Features

- Browser microphone capture through Streamlit WebRTC.
- Local inference using a pinned model revision; the application does not download model files itself.
- Live display of raw model scores and derived distress/nervousness indicators.
- Audio-quality warnings and a low-confidence state when the pipeline identifies quality concerns.
- A live trend view and indicative attention-category changes.
- A JSON call summary that excludes recorded audio.
- Model verification against the expected revision, required files, file hashes, and label mapping before recording is enabled.

## How it works

```text
Browser microphone
       |
       v
Short audio windows
       |
       v
Local, verified speech model
       |
       v
Raw label scores
       |
       v
Derived indicators + quality checks
       |
       v
Live human-review display
       |
       v
Stop recording -> audio-free JSON summary
```

The displayed indicators are derived from model outputs; they are not independent model classes or validated measurements of distress or nervousness.

## Requirements

- Windows is the documented setup environment.
- Python 3.12.
- Git, if you want to clone the repository.
- A local copy of the exact pinned model files (see below).
- A browser that permits microphone access.
- Dependencies listed in `requirements.txt`.

The application runs inference locally. The model files must already be present on your computer; AwazRakshak does not download them automatically. Internet access may still be needed to install Python packages or obtain the model files separately.

## Installation

Open PowerShell and clone the repository:

```powershell
git clone https://github.com/VIKINGvrc786/AwazRakshak.git
cd AwazRakshak
```

Create and activate a Python 3.12 virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, you can run the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configure the local model

AwazRakshak requires a **local copy of the exact pinned model revision**. The app does not fetch model files at runtime. In the app's **Local model path** field, enter one of the following:

1. The Hugging Face cache root containing the model cache.
2. The snapshot folder for the pinned revision.
3. A model directory containing `config.json`, `preprocessor_config.json`, and `model.safetensors`.

The app checks the pinned model metadata and the expected hashes for these files. Recording remains disabled if a file is missing, a hash differs, the label mapping is incorrect, or the revision does not match. Do not substitute a different model or revision without reviewing and documenting the change.

If your model is already cached in a non-default location, you can set the environment variable configured by the application before starting it. The app also accepts the path directly in the Local model path field.

## Run the application

From the repository folder, with the virtual environment installed:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL in the terminal, usually `http://localhost:8501`. Open that address in your browser.

## Use the app

1. Start AwazRakshak.
2. Enter the local model folder path, if it is not already configured.
3. Wait until the app confirms that the pinned model is verified.
4. Confirm that you are authorised to record and analyze the call by checking the authorisation box.
5. Select **Start recording** and allow microphone access in the browser.
6. Review live raw model scores, derived indicators, quality warnings, and trend information cautiously.
7. Select **Stop and save summary** when finished.
8. Review the summary and use **Download summary JSON** if you need a copy.

If the model is not verified or browser microphone capture is unavailable, the app displays a warning and recording cannot start.

## Privacy and saved data

Audio is held in a rolling in-memory buffer during the live session and is discarded when recording stops. The application saves a JSON summary—not the audio—to:

```text
.awazrakshak-data/summaries/
```

The summary includes session metadata such as timestamps, duration, number of analyzed windows, the final indicative category, peak derived indicators, quality flags, and the model ID/revision. The app also offers the summary as a browser download.

This design does not guarantee that audio is absent from every part of the surrounding system. Browser, operating-system, hosting, debugging, or third-party infrastructure may have separate behavior. Run locally on a trusted machine, obtain appropriate consent, protect saved summaries, and do not record calls unless you have the legal authority and consent required in your location. Do not commit recordings, summaries containing sensitive information, model weights, or credentials to the repository.

## Model details and limitations

AwazRakshak is a prototype, not a validated emotion-recognition or emergency-detection product.

- Scores are raw and uncalibrated; they should not be read as reliable probabilities.
- Distress and nervousness are derived indicators, not model output classes and not validated psychological measurements.
- Audio quality, language, accent, speaking style, background noise, microphone characteristics, and other factors may affect results.
- The model may produce scores for audio that does not reliably convey the intended signal.
- A high or changing indicator is not proof of distress, danger, deception, truthfulness, guilt, fraud, or intent.
- A low score does not establish that a person is safe or not distressed.
- Do not use the output alone to make emergency dispatch, triage, clinical, employment, legal, or other consequential decisions.

A trained and accountable human must interpret any signal in context and listen to the audio when appropriate. Do not present the app's output as an objective fact about the speaker.

## Tests

Run the available test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests that depend on model files may require the pinned model to be available locally. Do not treat passing software tests as evidence that the model is accurate, unbiased, calibrated, or suitable for real-world high-stakes use.

## Troubleshooting

**The app says the model is unavailable or invalid.** Check that the path points to the cache root, pinned snapshot, or model directory; that all required files exist; and that you have the exact expected revision and file contents. Do not bypass verification by swapping in another checkpoint.

**The browser does not capture audio.** Allow microphone permission for the local Streamlit page, check the operating-system microphone privacy settings, and confirm that the installed dependencies match `requirements.txt`.

**The Start recording button is disabled.** Confirm the authorisation checkbox is selected, the pinned model has passed verification, and WebRTC capture is available.

**PowerShell cannot activate the environment.** Use the explicit interpreter path shown in the installation instructions to run pip or Streamlit without activating the environment.

## Project structure

```text
AwazRakshak/
├── app.py             # Streamlit user interface and session summary
├── config.py          # Pinned model identity and indicator configuration
├── engine.py          # Local model loading and verification
├── live.py            # Live audio analysis pipeline
├── requirements.txt   # Pinned Python dependencies
├── tests/             # Automated tests
├── LICENSE            # MIT license for project source code
└── .awazrakshak-data/
    └── summaries/     # Locally saved audio-free JSON summaries
```

This is a high-level overview; the repository may contain additional modules and test files.

## License

### AwazRakshak source code

Copyright (c) 2026 VIKINGvrc786.

The original source code in this repository is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

The MIT License generally permits use, copying, modification, distribution, sublicensing, and sale of copies of the licensed software, provided the copyright and permission notices are included. The software is supplied “as is”, without warranty, as stated in the license.

### Third-party software and model files

The MIT license for this repository does **not** automatically apply to third-party Python packages, pretrained model weights, model documentation, or datasets. Those components may have separate licenses, attribution requirements, terms of use, or restrictions. Review the applicable notices and source terms before redistributing model files or using the project commercially. Do not assume that licensing the app's source code grants rights to redistribute the model or its training data.

## Project status

AwazRakshak is an experimental local prototype. Validate its behavior, privacy properties, robustness, and limitations independently before considering any broader deployment.
