"""Resolve and verify the pinned VoxTrace model with a safe synthetic pass."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EXPECTED_ID2LABEL, MODEL_ID, MODEL_OUTPUT_DIMENSION, MODEL_REVISION
from engine import (
    ModelVerificationError,
    _classifier_head_structure,
    validate_label_mapping,
    validate_score_vector,
)


def package_versions() -> dict[str, str | None]:
    names = (
        "numpy", "soundfile", "soxr", "torch", "transformers", "huggingface-hub",
        "safetensors", "streamlit", "webrtcvad-wheels",
    )
    result = {"python": sys.version.split()[0]}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_remote_revision() -> tuple[str | None, str | None]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(MODEL_ID)
        sha = str(info.sha)
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            return None, "Hub returned an invalid, non-immutable revision identifier"
        return sha, None
    except Exception as exc:  # Offline/network failures are recorded, not hidden.
        return None, f"{type(exc).__name__}: {exc}"


def inspect_cached_revisions() -> dict[str, Any]:
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == MODEL_ID:
                revisions = sorted(
                    {revision.commit_hash for revision in repo.revisions}
                )
                return {
                    "cache_repo_found": True,
                    "snapshot_revisions": revisions,
                    "configured_revision_present": MODEL_REVISION in revisions,
                }
    except Exception as exc:
        return {"cache_repo_found": False, "cache_error": f"{type(exc).__name__}: {exc}"}
    return {
        "cache_repo_found": False,
        "snapshot_revisions": [],
        "configured_revision_present": False,
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def verify(output_path: Path, *, offline: bool = False) -> dict[str, Any]:
    latest_revision = None
    latest_error = "offline flag requested; remote revision lookup not attempted"
    if not offline:
        latest_revision, latest_error = resolve_remote_revision()
    cache_info = inspect_cached_revisions()
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "verification_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "could_not_verify",
        "model_loaded": False,
        "human_approved": False,
        "model_id": MODEL_ID,
        "requested_pinned_revision": MODEL_REVISION,
        "current_available_revision": latest_revision,
        "current_revision_lookup_error": latest_error,
        "cache": cache_info,
        "environment": {
            "python": sys.version.split()[0],
            "package_versions": package_versions(),
        },
    }
    if not re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION):
        record["error"] = "Configured production revision is not a 40-character immutable SHA"
        output_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return record

    try:
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification

        snapshot_options = {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "allow_patterns": ["config.json", "preprocessor_config.json", "model.safetensors"],
        }
        if offline:
            snapshot_options["local_files_only"] = True
            snapshot = Path(snapshot_download(**snapshot_options))
        else:
            try:
                snapshot = Path(snapshot_download(**snapshot_options, local_files_only=True))
            except Exception:
                snapshot = Path(snapshot_download(**snapshot_options))
        if snapshot.name.lower() != MODEL_REVISION.lower():
            raise ModelVerificationError("Resolved local snapshot directory is not the requested SHA")

        common = {
            "revision": MODEL_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        config = AutoConfig.from_pretrained(MODEL_ID, **common)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, **common)
        model = AutoModelForAudioClassification.from_pretrained(
            MODEL_ID,
            use_safetensors=True,
            **common,
        )

        id2label = {int(key): str(value) for key, value in config.id2label.items()}
        labels = validate_label_mapping(
            config.id2label,
            config.label2id,
            config.num_labels,
            expected_id2label=EXPECTED_ID2LABEL,
        )
        model_labels = validate_label_mapping(
            model.config.id2label,
            model.config.label2id,
            model.config.num_labels,
            expected_id2label=EXPECTED_ID2LABEL,
        )
        if labels != model_labels:
            raise ModelVerificationError("Loaded model and config label maps disagree")

        config_hash = getattr(config, "_commit_hash", None)
        model_hash = getattr(model.config, "_commit_hash", None)
        feature_hash = getattr(feature_extractor, "_commit_hash", None)
        for observed in (config_hash, model_hash, feature_hash):
            if observed is not None and str(observed).lower() != MODEL_REVISION:
                raise ModelVerificationError("A loaded artifact reports a different revision SHA")
        if int(feature_extractor.sampling_rate) != 16_000:
            raise ModelVerificationError("Feature extractor does not require the configured 16 kHz rate")
        if int(model.config.num_labels) != MODEL_OUTPUT_DIMENSION:
            raise ModelVerificationError("Model configuration output dimension does not match the pin")

        model.eval().to("cpu")
        synthetic = np.zeros(16_000, dtype=np.float32)
        inputs = feature_extractor(synthetic, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits
        if logits.ndim != 2 or logits.shape[0] != 1 or int(logits.shape[-1]) != MODEL_OUTPUT_DIMENSION:
            raise ModelVerificationError(f"Synthetic forward pass returned logits shape {tuple(logits.shape)}")
        if not torch.isfinite(logits).all().item():
            raise ModelVerificationError("Synthetic forward pass returned non-finite logits")
        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
        raw_scores = [
            {"label": label, "score": score}
            for label, score in zip(labels, probabilities, strict=True)
        ]
        validated_scores = validate_score_vector(raw_scores, labels, MODEL_OUTPUT_DIMENSION)

        config_path = snapshot / "config.json"
        extractor_path = snapshot / "preprocessor_config.json"
        weights_path = snapshot / "model.safetensors"
        head_modules = _classifier_head_structure(model)
        feature_settings = feature_extractor.to_dict()
        record.update({
            "status": "verified",
            "model_loaded": True,
            "resolved_revision": MODEL_REVISION,
            "loaded_revision": MODEL_REVISION,
            "revision_confirmation": {
                "snapshot_directory_name": snapshot.name,
                "config_commit_hash": config_hash,
                "model_commit_hash": model_hash,
                "feature_extractor_commit_hash": feature_hash,
                "snapshot_directory_matches_requested_revision": snapshot.name == MODEL_REVISION,
                "online_current_revision_matches_pin": latest_revision == MODEL_REVISION,
            },
            "config": {
                "id2label": {str(key): value for key, value in id2label.items()},
                "label2id": {str(key): int(value) for key, value in config.label2id.items()},
                "num_labels": int(config.num_labels),
                "model_type": str(getattr(config, "model_type", "unknown")),
                "architectures": list(getattr(config, "architectures", []) or []),
                "hidden_size": getattr(config, "hidden_size", None),
            },
            "classifier_head": {
                "linear_modules": head_modules,
                "observed_output_dimension": int(logits.shape[-1]),
            },
            "feature_extractor": {
                "class": feature_extractor.__class__.__name__,
                "sampling_rate_hz": int(feature_extractor.sampling_rate),
                "settings": json_safe(feature_settings),
            },
            "synthetic_forward_pass": {
                "status": "passed",
                "input_description": "one second of synthetic digital-zero audio at 16 kHz",
                "logits_shape": [int(value) for value in logits.shape],
                "logits_finite": True,
                "all_class_scores_validated": True,
                "scores_by_label": {
                    item["label"]: item["score"] for item in validated_scores
                },
                "score_sum": sum(item["score"] for item in validated_scores),
            },
            "artifacts": {
                "config_json_sha256": sha256_file(config_path),
                "preprocessor_config_json_sha256": sha256_file(extractor_path),
                "model_safetensors_sha256": sha256_file(weights_path),
                "model_safetensors_bytes": weights_path.stat().st_size,
            },
            "safetensors_explicitly_requested": True,
            "pickle_training_artifacts_loaded": False,
            "raw_labels": list(labels),
        })
    except Exception as exc:
        record["status"] = "could_not_verify"
        record["error"] = f"{type(exc).__name__}: {exc}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "MODEL_VERIFICATION.json")
    parser.add_argument("--offline", action="store_true", help="Inspect the cache without checking the Hub")
    arguments = parser.parse_args()
    result = verify(arguments.output, offline=arguments.offline)
    print(json.dumps({
        "status": result.get("status"),
        "model_loaded": result.get("model_loaded"),
        "requested_pinned_revision": result.get("requested_pinned_revision"),
        "current_available_revision": result.get("current_available_revision"),
        "raw_labels": result.get("raw_labels"),
        "error": result.get("error"),
        "output": str(arguments.output.resolve()),
    }, indent=2, allow_nan=False))
    return 0 if result.get("status") == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
