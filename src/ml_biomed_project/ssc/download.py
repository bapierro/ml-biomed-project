"""Fetch only missing, checksum-pinned files for the public SSC4 demo release."""

import hashlib
import json
import shutil
import tempfile
import zipfile
from importlib import resources
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import get_session

from ml_biomed_project.ssc.inference import CHECKPOINT_SHA256


def _catalog():
    return json.loads(resources.files("ml_biomed_project.ssc").joinpath("demo.json").read_text())


def _path(folder, name):
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe demo path: {name!r}")
    path = folder.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(folder):
        raise ValueError(f"Demo path escapes its folder: {name!r}")
    return path


def _verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Invalid demo file size: {path}. Remove or archive it before retrying.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Invalid demo SHA256: {path}. Remove or archive it before retrying.")


def _temporary(parent):
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".ssc4-", dir=parent, delete=False) as stream:
        return Path(stream.name)


def _publish(temporary, destination, entry):
    _verify(temporary, entry)
    if destination.exists():
        _verify(destination, entry)
    else:
        temporary.replace(destination)


def _asset(catalog, entry, destination):
    temporary = _temporary(destination.parent)
    url = (
        f"https://github.com/{catalog['github_repository']}"
        f"/releases/download/{catalog['release']}/{entry['name']}"
    )
    try:
        headers = {"Accept": "application/octet-stream"}
        try:
            with get_session().stream(
                "GET", url, headers=headers, follow_redirects=True, timeout=60
            ) as response:
                response.raise_for_status()
                received = 0
                with temporary.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        received += len(chunk)
                        if received > entry["bytes"]:
                            raise ValueError("GitHub asset exceeds its catalog size")
                        stream.write(chunk)
        except Exception:
            raise RuntimeError(
                f"Could not download SSC4 asset {entry['name']}; check the connection and retry."
            ) from None
        _publish(temporary, destination, entry)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata(archive, folder, files, names):
    """Validate the entire ZIP directory before installing any missing member."""
    expected = {
        name
        for name, entry in files.items()
        if "crosspan_path" not in entry and not name.endswith("/checkpoint_final.pth")
    }
    with zipfile.ZipFile(archive) as bundle:
        seen = set()
        for info in bundle.infolist():
            _path(folder, info.filename)
            if (
                info.filename in seen
                or info.filename not in expected
                or info.is_dir()
                or (info.external_attr >> 16) & 0o170000 == 0o120000
                or info.file_size != files[info.filename]["bytes"]
            ):
                raise ValueError(f"Unexpected, duplicate, or invalid ZIP member: {info.filename!r}")
            seen.add(info.filename)
        if seen != expected:
            raise ValueError("SSC4 metadata archive does not match the catalog")
        for name in names:
            destination = _path(folder, name)
            temporary = _temporary(destination.parent)
            try:
                with bundle.open(name) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                _publish(temporary, destination, files[name])
            finally:
                temporary.unlink(missing_ok=True)


def ensure_demo(folder):
    """Return the verified demo folder, downloading missing files only.

    Public release files need no GitHub login. Original images and masks come
    directly from the catalog's pinned CrossPan revision. Existing corrupt files
    are rejected, never silently repaired or overwritten.
    """
    folder = Path(folder).expanduser().resolve()
    catalog = _catalog()
    checkpoint = catalog["checkpoint"]
    files = catalog["files"]
    if (
        catalog["version"] != 1
        or catalog["github_repository"] != "bapierro/ml-biomed-project"
        or catalog["release"] != "ssc4-demo-v1"
        or checkpoint["sha256"] != CHECKPOINT_SHA256
        or files[checkpoint["path"]] != {key: checkpoint[key] for key in ("sha256", "bytes")}
        or catalog["crosspan"]
        != {
            "repo_id": "linkai-peng/CrossPan",
            "revision": "0da3821e8af2dfa4c22d56e282372614cb426227",
        }
    ):
        raise ValueError("Unsupported or inconsistent SSC4 demo catalog")
    missing = []
    for name, entry in files.items():
        path = _path(folder, name)
        if path.exists():
            _verify(path, entry)
        else:
            missing.append(name)
    if not missing:
        return folder

    metadata = [
        name
        for name in missing
        if name != checkpoint["path"] and "crosspan_path" not in files[name]
    ]
    if checkpoint["path"] in missing or metadata:
        if checkpoint["path"] in missing:
            _asset(catalog, checkpoint, _path(folder, checkpoint["path"]))
        if metadata:
            archive = _temporary(folder)
            archive.unlink()
            try:
                _asset(catalog, catalog["metadata"], archive)
                _metadata(archive, folder, files, metadata)
            finally:
                archive.unlink(missing_ok=True)

    for name in missing:
        entry = files[name]
        if "crosspan_path" not in entry:
            continue
        cached = Path(
            hf_hub_download(
                repo_id=catalog["crosspan"]["repo_id"],
                repo_type="dataset",
                revision=catalog["crosspan"]["revision"],
                filename=entry["crosspan_path"],
            )
        )
        _verify(cached, entry)
        destination = _path(folder, name)
        temporary = _temporary(destination.parent)
        try:
            shutil.copyfile(cached, temporary)
            _publish(temporary, destination, entry)
        finally:
            temporary.unlink(missing_ok=True)
    return folder
