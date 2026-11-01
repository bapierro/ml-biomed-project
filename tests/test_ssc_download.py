"""Missing-file downloads use fake bytes and sessions, never network or models."""

import hashlib
import io
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

from ml_biomed_project.ssc import download


def entry(payload):
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def archive_bytes(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    return stream.getvalue()


@pytest.fixture
def bundle(monkeypatch):
    payloads = {
        "model/fold_0/checkpoint_final.pth": b"fake checkpoint, never unpickled",
        "manifest.json": b"fake manifest",
        "cases/example/ssc4.nii.gz": b"fake prediction",
        "cases/example/image.nii.gz": b"fake source image",
        "cases/example/reference.nii.gz": b"fake source reference",
    }
    files = {name: entry(payload) for name, payload in payloads.items()}
    for name, repo_path in [
        ("cases/example/image.nii.gz", "T1/val/images/EXAMPLE_T1_0000.nii.gz"),
        ("cases/example/reference.nii.gz", "T1/val/labels/EXAMPLE_T1.nii.gz"),
    ]:
        files[name]["crosspan_path"] = repo_path
    checkpoint = "model/fold_0/checkpoint_final.pth"
    metadata = archive_bytes(
        [(name, payloads[name]) for name in ("manifest.json", "cases/example/ssc4.nii.gz")]
    )
    catalog = {
        "version": 1,
        "github_repository": "bapierro/ml-biomed-project",
        "release": "ssc4-demo-v1",
        "checkpoint": {
            **files[checkpoint],
            "asset_id": 1,
            "name": "ssc4-final.pth",
            "path": checkpoint,
        },
        "metadata": {**entry(metadata), "asset_id": 2, "name": "ssc4-demo-v1.zip"},
        "crosspan": {
            "repo_id": "linkai-peng/CrossPan",
            "revision": "0da3821e8af2dfa4c22d56e282372614cb426227",
        },
        "files": files,
    }
    monkeypatch.setattr(download, "_catalog", lambda: catalog)
    monkeypatch.setattr(download, "CHECKPOINT_SHA256", files[checkpoint]["sha256"])
    return catalog, payloads, metadata


def stage(folder, payloads, exclude=()):
    for name, payload in payloads.items():
        if name not in exclude:
            destination = folder / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected process or network")


def test_complete_cache_is_offline(tmp_path, monkeypatch, bundle):
    _, payloads, _ = bundle
    stage(tmp_path, payloads)
    for name in ("get_session", "hf_hub_download"):
        monkeypatch.setattr(download, name, forbidden)
    assert download.ensure_demo(tmp_path) == tmp_path


def test_missing_mri_only_uses_exact_pinned_hf_and_preserves_cache(tmp_path, monkeypatch, bundle):
    catalog, payloads, _ = bundle
    name = "cases/example/image.nii.gz"
    stage(tmp_path, payloads, exclude=[name])
    cached = tmp_path / "hf-cache"
    cached.write_bytes(payloads[name])
    calls = []

    def hf(**kwargs):
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setattr(download, "get_session", forbidden)
    monkeypatch.setattr(download, "hf_hub_download", hf)
    download.ensure_demo(tmp_path)
    assert calls == [
        {
            "repo_id": catalog["crosspan"]["repo_id"],
            "repo_type": "dataset",
            "revision": catalog["crosspan"]["revision"],
            "filename": catalog["files"][name]["crosspan_path"],
        }
    ]
    assert (tmp_path / name).read_bytes() == cached.read_bytes() == payloads[name]
    assert (tmp_path / name).stat().st_ino != cached.stat().st_ino


def test_corrupt_existing_file_stops_before_network(tmp_path, monkeypatch, bundle):
    _, payloads, _ = bundle
    stage(tmp_path, payloads)
    destination = tmp_path / "manifest.json"
    destination.write_bytes(b"corrupt")
    monkeypatch.setattr(download, "get_session", forbidden)
    monkeypatch.setattr(download, "hf_hub_download", forbidden)
    with pytest.raises(ValueError, match="Invalid demo"):
        download.ensure_demo(tmp_path)
    assert destination.read_bytes() == b"corrupt"


def test_public_assets_are_anonymous_verified_and_only_missing_metadata_installed(
    tmp_path, monkeypatch, bundle
):
    catalog, payloads, metadata = bundle
    checkpoint = catalog["checkpoint"]["path"]
    stage(tmp_path, payloads, exclude=[checkpoint, "manifest.json"])
    existing = tmp_path / "cases/example/ssc4.nii.gz"
    previous = existing.stat().st_mtime_ns
    calls = []
    assets = {"ssc4-final.pth": payloads[checkpoint], "ssc4-demo-v1.zip": metadata}

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield self.payload[:5]
            yield self.payload[5:]

    def stream(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return Response(assets[url.rsplit("/", 1)[1]])

    monkeypatch.setenv("GH_TOKEN", "unused-token")
    monkeypatch.setenv("GITHUB_TOKEN", "also-unused")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(download, "get_session", lambda: SimpleNamespace(stream=stream))
    monkeypatch.setattr(download, "hf_hub_download", forbidden)
    download.ensure_demo(tmp_path)
    assert len(calls) == 2
    for method, url, kwargs in calls:
        assert method == "GET"
        assert url.startswith(
            "https://github.com/bapierro/ml-biomed-project/releases/download/ssc4-demo-v1/"
        )
        assert kwargs == {
            "headers": {
                "Accept": "application/octet-stream",
            },
            "follow_redirects": True,
            "timeout": 60,
        }
    assert existing.stat().st_mtime_ns == previous
    for name, payload in payloads.items():
        assert (tmp_path / name).read_bytes() == payload
    assert not list(tmp_path.rglob(".ssc4-*"))


@pytest.mark.parametrize("bad_name", ["../escape", "/absolute", "unexpected.json", "manifest.json"])
def test_unsafe_or_duplicate_zip_rejected_before_mutation(tmp_path, bundle, bad_name):
    catalog, payloads, _ = bundle
    members = [("manifest.json", payloads["manifest.json"]), (bad_name, b"bad")]
    archive = tmp_path / "download.zip"
    with pytest.warns(UserWarning) if bad_name == "manifest.json" else _no_warning():
        archive.write_bytes(archive_bytes(members))
    with pytest.raises(ValueError):
        download._metadata(archive, tmp_path, catalog["files"], ["manifest.json"])
    assert not (tmp_path / "manifest.json").exists()


def _no_warning():
    from contextlib import nullcontext

    return nullcontext()


def test_member_checksum_failure_never_publishes_corrupt_file(tmp_path, bundle):
    catalog, payloads, _ = bundle
    archive = tmp_path / "download.zip"
    corrupt = b"X" * len(payloads["manifest.json"])
    archive.write_bytes(
        archive_bytes(
            [
                ("manifest.json", corrupt),
                ("cases/example/ssc4.nii.gz", payloads["cases/example/ssc4.nii.gz"]),
            ]
        )
    )
    with pytest.raises(ValueError, match="SHA256"):
        download._metadata(archive, tmp_path, catalog["files"], ["manifest.json"])
    assert not (tmp_path / "manifest.json").exists()
