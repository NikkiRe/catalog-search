"""A corrupted or unsafe release must never become a usable installation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


spec = importlib.util.spec_from_file_location("download_demo", Path(__file__).parents[1] / "scripts/download_demo.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _bundle(tmp_path, name="encoder.joblib"):
    archive = tmp_path / "demo.zip"
    value = b"synthetic-test-payload-not-a-pickle"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, value)
        stream.writestr("artifact_checksums.json", json.dumps({name: hashlib.sha256(value).hexdigest()}))
    return {"url": archive.as_uri(), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}


def test_verified_archive_installs_and_preserves_existing_output(tmp_path):
    manifest = _bundle(tmp_path)
    output = module.install(manifest, tmp_path / "artifacts")
    assert (output / "encoder.joblib").read_bytes().startswith(b"synthetic-test")
    with pytest.raises(FileExistsError):
        module.install(manifest, output)


def test_corrupt_release_stops_before_installation(tmp_path):
    manifest = _bundle(tmp_path)
    manifest["sha256"] = "0" * 64
    output = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="checksum"):
        module.install(manifest, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["../escape.joblib", "/absolute.joblib", "..\\escape.joblib"])
def test_archive_path_cannot_escape_destination(tmp_path, name):
    manifest = _bundle(tmp_path, name)
    with pytest.raises(ValueError, match="Unsafe"):
        module.install(manifest, tmp_path / "artifacts")
    assert not (tmp_path / "escape.joblib").exists()
