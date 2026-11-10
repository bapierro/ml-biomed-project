"""Audit local MSD brain T1/T2 pairs without changing images or fitting a model."""

import argparse
import hashlib
import json
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import psutil


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def channel_stats(array):
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite MRI intensities")
    mask = array != 0
    count = int(mask.sum())
    if not count or float(np.std(array[mask])) == 0:
        raise ValueError("Empty or constant MRI channel")
    bbox = []
    for axis in range(3):
        occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        bbox.append([int(occupied[0]), int(occupied[-1] + 1)])
    values = array[mask]
    # Bounded deterministic subsample for descriptive quantiles, never a normalization fit.
    sample = values[:: max(1, len(values) // 100_000)]
    return {
        "array_sha256": array_digest(array),
        "nonzero_voxels": count,
        "nonzero_bbox": bbox,
        "nonzero_mean": float(values.mean()),
        "nonzero_std": float(values.std()),
        "range": [float(array.min()), float(array.max())],
        "sample_quantiles_01_50_99": np.quantile(sample, [0.01, 0.5, 0.99]).tolist(),
    }


def inspect_case(root, case):
    image_path = root / "imagesTr" / f"{case}.nii.gz"
    label_path = root / "labelsTr" / f"{case}.nii.gz"
    image, label = nib.load(image_path), nib.load(label_path)
    if len(image.shape) != 4 or image.shape[-1] != 4 or image.shape[:3] != label.shape:
        raise ValueError(f"Unexpected image/label shapes: {case}")
    if not np.allclose(image.affine, label.affine, atol=1e-5, rtol=0):
        raise ValueError(f"Image/reference grid mismatch: {case}")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-9:
        raise ValueError(f"Invalid affine: {case}")
    mask = np.asanyarray(label.dataobj)
    values, counts = np.unique(mask, return_counts=True)
    if not set(values.tolist()) <= {0, 1, 2, 3} or not np.any(mask > 0):
        raise ValueError(f"Invalid/empty tumor labels: {case}")
    # One read of each source file. Other modalities are not used for statistics or fitting.
    data = np.asanyarray(image.dataobj)
    t1, t2 = channel_stats(data[..., 1]), channel_stats(data[..., 3])
    return {
        "case_id": case,
        "image_sha256": digest(image_path),
        "label_sha256": digest(label_path),
        "image_bytes": image_path.stat().st_size,
        "label_bytes": label_path.stat().st_size,
        "shape": list(image.shape),
        "spacing": [float(x) for x in image.header.get_zooms()[:3]],
        "affine": image.affine.tolist(),
        "spatial_unit": image.header.get_xyzt_units()[0],
        "label_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
        "label_array_sha256": array_digest(mask),
        "whole_tumor_voxels": int((mask > 0).sum()),
        "T1": t1,
        "T2": t2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior audits; choose a new output directory")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    record = {"status": "INCOMPLETE", "cases_completed": 0}
    rows = []

    def timeout(signum, frame):
        raise TimeoutError("Bounded data audit deadline")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.deadline_seconds)
    try:
        metadata = json.loads((args.dataset / "dataset.json").read_text())
        if metadata["modality"].get("1") != "T1w" or metadata["modality"].get("3") != "T2w":
            raise ValueError("Unverified T1w/T2w channel order")
        cases = sorted(
            Path(row["image"]).name.removesuffix(".nii.gz") for row in metadata["training"]
        )
        if len(cases) != metadata["numTraining"] or len(set(cases)) != len(cases):
            raise ValueError("Training catalog count/uniqueness mismatch")
        for folder in ["imagesTr", "labelsTr"]:
            if {
                p.name.removesuffix(".nii.gz") for p in (args.dataset / folder).glob("*.nii.gz")
            } != set(cases):
                raise ValueError(f"Local {folder} coverage differs from official catalog")
        with (args.output / "cases.jsonl").open("w") as stream:
            for case in cases:
                if psutil.Process().memory_info().rss > 2_000_000_000:
                    raise MemoryError("Data audit exceeded 2 GB RSS")
                row = inspect_case(args.dataset, case)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                record["cases_completed"] = len(rows)
                if len(rows) % 25 == 0:
                    print(f"Audited {len(rows)}/{len(cases)} cases", flush=True)
        duplicates = {}
        for channel in ["T1", "T2"]:
            hashes = defaultdict(list)
            for row in rows:
                hashes[row[channel]["array_sha256"]].append(row["case_id"])
            duplicates[channel] = [v for v in hashes.values() if len(v) > 1]
        record.update(
            status="COMPLETE",
            catalog_sha256=digest(args.dataset / "dataset.json"),
            cases_sha256=digest(args.output / "cases.jsonl"),
            shapes=dict(Counter(str(r["shape"]) for r in rows)),
            spacings=dict(Counter(str(r["spacing"]) for r in rows)),
            spatial_units=dict(Counter(r["spatial_unit"] for r in rows)),
            exact_channel_duplicates=duplicates,
            unlabeled_test_count=len(list((args.dataset / "imagesTs").glob("*.nii.gz"))),
            unlabeled_test_arrays_read=0,
            source_bytes=sum(r["image_bytes"] + r["label_bytes"] for r in rows),
            scope="Model-blind T1/T2 geometry, finite values, labels and exact duplicates; not biological identity verification",
        )
    except BaseException as error:
        record.update(status="FAILED", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        record["elapsed_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
