"""Check source selection and file integrity without MRI, preprocessing or training."""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def prepare():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_crosspan.py"
    spec = importlib.util.spec_from_file_location("prepare_crosspan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path, prepare, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    origin = tmp_path / "release"
    records = []
    for case, split in [("A_T1", "train"), ("B_T1", "train"), ("C_T1", "val")]:
        row = {"case_id": case, "split": split}
        for key, folder, suffix in [("image", "images", "_0000"), ("label", "labels", "")]:
            path = Path(f"T1/{split}/{folder}/{case}{suffix}.nii.gz")
            payload = (case + key).encode()
            (origin / path).parent.mkdir(parents=True, exist_ok=True)
            (origin / path).write_bytes(payload)
            row[key] = str(path)
            row[key + "_sha256"] = hashlib.sha256(payload).hexdigest()
        records.append(row)
    with (reference / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (reference / "splits_final.json").write_text(
        json.dumps([{"train": ["A_T1", "B_T1"], "val": ["C_T1"]}])
    )
    (reference / "SSC2D.json").write_text(json.dumps({"plans_name": "SSC2D"}))
    (reference / "dataset_fingerprint.json").write_text(json.dumps({"spacings": [[3, 1, 1]]}))
    (reference / "source.json").write_text(
        json.dumps(
            {
                "dataset": "linkai-peng/CrossPan",
                "sequence": "T1",
                "revision": "0da3821e8af2dfa4c22d56e282372614cb426227",
            }
        )
    )
    monkeypatch.setattr(prepare, "REFERENCE", reference)
    return origin, tmp_path / "workspace"


def test_official_catalog_is_source_only_and_preserves_alias_exclusion(prepare):
    with (prepare.REFERENCE / "cases.csv").open() as stream:
        cases = list(csv.DictReader(stream))
    split = json.loads((prepare.REFERENCE / "splits_final.json").read_text())[0]
    assert len(cases) == 369
    assert (len(split["train"]), len(split["val"])) == (323, 46)
    assert set(split["train"]).isdisjoint(split["val"])
    assert "MCF_T1_0090" not in split["train"]
    assert "MCF_T1_0067" in split["val"]
    assert all(r["image"].startswith(f"T1/{r['split']}/images/") for r in cases)
    assert set(r["case_id"] for r in cases) == set(split["train"]) | set(split["val"])


def test_stage_keeps_val_out_of_training_split_and_ignores_unlisted_targets(prepare, source):
    origin, workspace = source
    (origin / "T2").mkdir()
    (origin / "T2/unlisted.nii.gz").write_bytes(b"not source")
    assert prepare.stage(origin, workspace) == (2, 1)
    raw = workspace / "raw" / prepare.DATASET
    pre = workspace / "preprocessed" / prepare.DATASET
    assert len(list((raw / "imagesTr").glob("*.nii.gz"))) == 3
    assert not list(workspace.rglob("unlisted*"))
    assert json.loads((pre / "splits_final.json").read_text())[0]["val"] == ["C_T1"]
    assert json.loads((raw / "dataset.json").read_text())["numTraining"] == 3
    # Native on_train_start copies this file even when preprocessing already exists.
    assert (pre / "dataset_fingerprint.json").is_file()
    # Re-running reuses verified destination files, even if the release folder moves.
    assert prepare.stage(origin / "moved", workspace) == (2, 1)


def test_corrupt_file_fails_without_silent_repair(prepare, source):
    origin, workspace = source
    prepare.stage(origin, workspace)
    bad = workspace / "raw" / prepare.DATASET / "labelsTr/A_T1.nii.gz"
    bad.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare.stage(origin, workspace)
    assert bad.read_bytes() == b"corrupt"


def test_extra_staged_case_cannot_enter_fit(prepare, source):
    origin, workspace = source
    prepare.stage(origin, workspace)
    extra = workspace / "raw" / prepare.DATASET / "imagesTr/T2_0000.nii.gz"
    extra.write_bytes(b"target")
    with pytest.raises(ValueError, match="Unexpected cases"):
        prepare.stage(origin, workspace)


def test_preprocessing_delegates_to_package_with_one_worker(prepare, tmp_path, monkeypatch):
    calls = []
    api = SimpleNamespace(preprocess_dataset=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "nnunetv2.experiment_planning.plan_and_preprocess_api", api)
    monkeypatch.setattr(prepare.os, "environ", prepare.os.environ.copy())
    prepare.preprocess(tmp_path)
    assert calls == [
        ((801,), {"plans_identifier": "SSC2D", "configurations": ("2d",), "num_processes": [1]})
    ]
    assert prepare.os.environ["nnUNet_raw"] == str(tmp_path / "raw")


def test_download_uses_exact_full_catalog_allowlist_and_pinned_revision(
    prepare, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kw: calls.append(kw)),
    )
    assert prepare.download_missing(tmp_path / "source", tmp_path / "workspace") == 738
    cases, _, info = prepare.read_catalog()
    assert calls == [
        {
            "repo_id": "linkai-peng/CrossPan",
            "repo_type": "dataset",
            "revision": info["revision"],
            "allow_patterns": [r[k] for r in cases for k in ("image", "label")],
            "local_dir": str(tmp_path / "source"),
            "max_workers": 2,
        }
    ]
    assert all(p.startswith(("T1/train/", "T1/val/")) for p in calls[0]["allow_patterns"])


def test_download_cli_defaults_source_and_verifies_fake_downloads(
    prepare, source, monkeypatch, tmp_path
):
    _, workspace = source
    calls = []
    cases, _, _ = prepare.read_catalog()
    payloads = {r[k]: (r["case_id"] + k).encode() for r in cases for k in ("image", "label")}

    def fake_download(**kwargs):
        calls.append(kwargs)
        for path in kwargs["allow_patterns"]:
            destination = Path(kwargs["local_dir"]) / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payloads[path])

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_download)
    )
    monkeypatch.chdir(tmp_path)
    prepare.main(["--download", "--stage-only", "--workspace", str(workspace)])
    assert calls[0]["local_dir"] == str(tmp_path / "data/crosspan")
    assert len(calls[0]["allow_patterns"]) == 6
    assert (
        workspace / "raw" / prepare.DATASET / "imagesTr/A_T1_0000.nii.gz"
    ).read_bytes() == b"A_T1image"


def test_cached_source_and_staged_files_skip_network(prepare, source, monkeypatch):
    origin, workspace = source

    def forbidden(**kwargs):
        raise AssertionError("Cached files triggered network")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=forbidden)
    )
    assert prepare.download_missing(origin, workspace) == 0
    prepare.stage(origin, workspace)
    assert prepare.download_missing(origin / "moved", workspace) == 0
    assert prepare.stage(origin / "moved", workspace) == (2, 1)


def test_download_only_missing_subset_reusing_staged_destination(prepare, source, monkeypatch):
    origin, workspace = source
    prepare.stage(origin, workspace)
    (origin / "T1/train/images/A_T1_0000.nii.gz").unlink()  # Staged copy is enough.
    missing = "T1/train/images/B_T1_0000.nii.gz"
    (origin / missing).unlink()
    (workspace / "raw" / prepare.DATASET / "imagesTr/B_T1_0000.nii.gz").unlink()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / missing
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"B_T1image")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_download)
    )
    assert prepare.download_missing(origin, workspace) == 1
    assert calls[0]["allow_patterns"] == [missing]
    assert prepare.stage(origin, workspace) == (2, 1)


@pytest.mark.parametrize("location", ["source", "destination"])
def test_corrupt_present_file_rejected_before_network(prepare, source, monkeypatch, location):
    origin, workspace = source
    if location == "destination":
        prepare.stage(origin, workspace)
        bad = workspace / "raw" / prepare.DATASET / "labelsTr/A_T1.nii.gz"
    else:
        bad = origin / "T1/train/labels/A_T1.nii.gz"
    bad.write_bytes(b"corrupt")

    def forbidden(**kwargs):
        raise AssertionError("Corrupt file reached network")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=forbidden)
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare.main([str(origin), "--download", "--stage-only", "--workspace", str(workspace)])
    assert bad.read_bytes() == b"corrupt"


def test_missing_source_requires_download_flag(prepare):
    with pytest.raises(SystemExit) as error:
        prepare.main(["--stage-only"])
    assert error.value.code == 2


def test_downloader_missing_outputs_cannot_be_marked_prepared(
    prepare, source, monkeypatch, tmp_path
):
    _, workspace = source
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **kw: None)
    )
    with pytest.raises(FileNotFoundError, match="Missing"):
        prepare.main(
            [str(tmp_path / "absent"), "--download", "--stage-only", "--workspace", str(workspace)]
        )


@pytest.mark.parametrize(
    "invalid_path", ["T2/test/images/A_T1_0000.nii.gz", "T1/test/images/A_T1_0000.nii.gz"]
)
def test_target_paths_rejected_before_downloader(prepare, source, monkeypatch, invalid_path):
    origin, workspace = source
    catalog = prepare.REFERENCE / "cases.csv"
    rows = list(csv.DictReader(catalog.open()))
    rows[0]["image"] = invalid_path
    with catalog.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    def forbidden(**kwargs):
        raise AssertionError("Target path reached network")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=forbidden)
    )
    with pytest.raises(ValueError, match="forbidden source path"):
        prepare.download_missing(origin, workspace)
