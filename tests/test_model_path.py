import engine
import pytest
from config import MODEL_ID, MODEL_REVISION


@pytest.mark.parametrize("cache_subfolder", ["", "hub"])
def test_hugging_face_cache_root_selects_pinned_snapshot(
    tmp_path,
    monkeypatch,
    cache_subfolder,
):
    cache_root = tmp_path / cache_subfolder if cache_subfolder else tmp_path
    repo_folder = cache_root / f"models--{MODEL_ID.replace('/', '--')}"
    snapshot = repo_folder / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        engine,
        "_verified_local_model_path",
        lambda path, revision: str(path),
    )

    assert engine._verified_snapshot_path(MODEL_REVISION, tmp_path) == str(snapshot)


def test_local_path_without_pinned_snapshot_fails_closed(tmp_path):
    available, reason = engine.pinned_model_cache_status(tmp_path)

    assert not available
    assert "no snapshot for the pinned model revision" in reason


def test_local_snapshot_still_requires_manifest_and_exact_file_hashes(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    available, reason = engine.pinned_model_cache_status(snapshot)

    assert not available
    assert "does not match the verified pin" in reason
