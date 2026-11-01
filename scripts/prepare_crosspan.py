"""Prepare the fixed CrossPan T1 development split with native nnU-Net preprocessing.

Reads a local release or downloads the exact pinned T1 TRAIN/VAL catalog on request.
No training or label repair. TEST, T2 and unlisted release files are never selected.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / "configs/crosspan-t1"
DATASET = "Dataset801_CrossPanT1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_metadata(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Existing metadata differs: {path}. Use a separate workspace.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_catalog():
    """Read one validated source-only allowlist, split and pinned repository identity."""
    with (REFERENCE / "cases.csv").open(newline="") as stream:
        cases = list(csv.DictReader(stream))
    splits = json.loads((REFERENCE / "splits_final.json").read_text())
    info = json.loads((REFERENCE / "source.json").read_text())
    if (
        info.get("dataset") != "linkai-peng/CrossPan"
        or info.get("sequence") != "T1"
        or not re.fullmatch(r"[0-9a-f]{40}", info.get("revision", ""))
    ):
        raise ValueError("Pinned CrossPan T1 source identity required.")
    seen = set()
    for row in cases:
        case, split = row["case_id"], row["split"]
        if (
            split not in {"train", "val"}
            or not re.fullmatch(r"[A-Za-z0-9_-]+", case)
            or case in seen
        ):
            raise ValueError("Catalog must contain unique source TRAIN/VAL cases only.")
        seen.add(case)
        for kind, folder, suffix in [("image", "images", "_0000"), ("label", "labels", "")]:
            if row[kind] != f"T1/{split}/{folder}/{case}{suffix}.nii.gz" or not re.fullmatch(
                r"[0-9a-f]{64}", row[kind + "_sha256"]
            ):
                raise ValueError("Catalog contains a forbidden source path or invalid checksum.")
    expected = [
        {
            "train": sorted(r["case_id"] for r in cases if r["split"] == "train"),
            "val": sorted(r["case_id"] for r in cases if r["split"] == "val"),
        }
    ]
    if not cases or splits != expected or set(expected[0]["train"]) & set(expected[0]["val"]):
        raise ValueError("Catalog and official source split disagree.")
    return cases, splits, info


def catalog_files(cases, source, workspace):
    raw = Path(workspace).resolve() / "raw" / DATASET
    for row in cases:
        for kind, folder, suffix in [
            ("image", "imagesTr", "_0000.nii.gz"),
            ("label", "labelsTr", ".nii.gz"),
        ]:
            yield (
                row,
                kind,
                Path(source).resolve() / row[kind],
                raw / folder / (row["case_id"] + suffix),
            )


def missing_files(cases, source, workspace):
    """Verify EVERY present source/destination before any network; never repair corrupt files."""
    raw = Path(workspace).resolve() / "raw" / DATASET
    expected_names = {
        "imagesTr": {r["case_id"] + "_0000.nii.gz" for r in cases},
        "labelsTr": {r["case_id"] + ".nii.gz" for r in cases},
    }
    for folder, names in expected_names.items():
        extras = {f.name for f in (raw / folder).glob("*.nii.gz")} - names
        if extras:
            raise ValueError(f"Unexpected cases in {raw / folder}: {sorted(extras)}")
    missing = []
    for row, kind, origin, destination in catalog_files(cases, source, workspace):
        for candidate in (origin, destination):
            if candidate.exists() and (
                not candidate.is_file() or digest(candidate) != row[kind + "_sha256"]
            ):
                raise ValueError(f"Pinned release checksum mismatch: {candidate}")
        if not origin.is_file() and not destination.is_file():
            missing.append(row[kind])
    return missing


def download_missing(source, workspace, catalog=None):
    """Use Hugging Face's downloader ONLY for exact catalog paths not already verified locally."""
    cases, _, info = catalog or read_catalog()
    missing = missing_files(cases, source, workspace)
    if missing:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=info["dataset"],
            repo_type="dataset",
            revision=info["revision"],
            allow_patterns=missing,
            local_dir=str(Path(source).resolve()),
            max_workers=2,
        )
    return len(missing)


def stage(source, workspace, catalog=None):
    """Copy the catalog's exact bytes and official split; reuse verified local copies."""
    source, workspace = Path(source).resolve(), Path(workspace).resolve()
    raw = workspace / "raw" / DATASET
    pre = workspace / "preprocessed" / DATASET
    cases, splits, _ = catalog or read_catalog()
    missing = missing_files(cases, source, workspace)
    if missing:
        raise FileNotFoundError(
            f"Missing {source / missing[0]}. Supply the pinned T1 files or use --download."
        )
    for _, _, origin, destination in catalog_files(cases, source, workspace):
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
    dataset = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "pancreas": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }
    write_metadata(raw / "dataset.json", dataset)
    write_metadata(pre / "dataset.json", dataset)
    write_metadata(pre / "SSC2D.json", json.loads((REFERENCE / "SSC2D.json").read_text()))
    write_metadata(pre / "splits_final.json", splits)
    write_metadata(
        pre / "dataset_fingerprint.json",
        json.loads((REFERENCE / "dataset_fingerprint.json").read_text()),
    )
    (workspace / "results").mkdir(parents=True, exist_ok=True)
    return len(splits[0]["train"]), len(splits[0]["val"])


def preprocess(workspace):
    workspace = Path(workspace).resolve()
    os.environ.update(
        {
            f"nnUNet_{name}": str(workspace / folder)
            for name, folder in [
                ("raw", "raw"),
                ("preprocessed", "preprocessed"),
                ("results", "results"),
            ]
        }
    )
    os.environ.update(
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="2"
    )
    # Imports follow path setup: nnU-Net resolves these environment variables on import.
    from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess_dataset

    preprocess_dataset(801, plans_identifier="SSC2D", configurations=("2d",), num_processes=[1])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        help="Local release folder; --download defaults to data/crosspan",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download only missing pinned T1 TRAIN/VAL files"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("data/nnunet"),
        help="Local nnU-Net raw/preprocessed/results folder",
    )
    parser.add_argument(
        "--stage-only", action="store_true", help="Copy files/metadata without preprocessing"
    )
    args = parser.parse_args(argv)
    if args.source is None and not args.download:
        parser.error("source is required unless --download is supplied")
    source = args.source if args.source is not None else Path("data/crosspan")
    catalog = read_catalog()
    if args.download:
        download_missing(source, args.workspace, catalog)
    train, val = stage(source, args.workspace, catalog)
    print(f"Prepared {train} T1 training / {val} development-validation cases.", flush=True)
    if not args.stage_only:
        preprocess(args.workspace)
    print(f"Workspace: {args.workspace.resolve()}")
    print("Next: python scripts/train_ssc.py 801")


if __name__ == "__main__":
    main()
