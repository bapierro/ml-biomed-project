"""Prepare one frozen MSD brain split for shared native RAW/SSC4 training.

Run stage, plan, pilot, then preprocess as separate guarded phases. Original
four-channel images are read-only; only T1 (channel 1) enters the training cache.
Each phase keeps its receipt, including failures, and refuses an automatic retry.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import time
from contextlib import contextmanager
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
):
    os.environ[_name] = "2"

# Native numeric libraries must see the thread cap before their first import.
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

DATASET = "Dataset802_BrainT1WT"
PLANS = "Brain2D"
MIN_FREE = 12_000_000_000
MODEL_AND_EVALUATION_RESERVE = 2_300_000_000


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_manifest(path, expected_counts=(387, 97)):
    path = Path(path).resolve()
    rows = read_json(path)["rows"]
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"BRATS_[0-9]+", x) for x in ids):
        raise ValueError("Manifest case IDs must be unique BRATS identifiers")
    counts = tuple(sum(row["role"] == role for row in rows) for role in ("train", "val"))
    if counts != expected_counts or sum(counts) != len(rows):
        raise ValueError("Manifest must preserve the frozen train/val allocation")
    for row in rows:
        for key in ("image", "label"):
            source = Path(row[key])
            if not source.is_absolute() or source.name != row["case_id"] + ".nii.gz":
                raise ValueError("Manifest requires absolute original case paths")
            if not re.fullmatch(r"[0-9a-f]{64}", row[key + "_sha256"]):
                raise ValueError("Manifest requires source SHA256 values")
    return sorted(rows, key=lambda row: row["case_id"])


class Guard:
    def __init__(self, workspace, seconds, stop_files=(), minimum_free=MIN_FREE):
        self.workspace = Path(workspace)
        self.deadline = time.monotonic() + seconds
        self.stop_files = [self.workspace / "STOP_REQUESTED", *map(Path, stop_files)]
        self.minimum_free = minimum_free

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Preparation deadline reached")
        if any(path.exists() for path in self.stop_files):
            raise InterruptedError("Preparation STOP marker present")
        existing = self.workspace
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        if free < self.minimum_free:
            raise OSError("Preparation disk floor reached")
        return free


def spatial_copy(array, original, destination):
    """Write exact selected values on the original grid, preserving qform/sform."""
    header = original.header.copy()
    header.set_data_dtype(array.dtype)
    output = nib.Nifti1Image(array, original.affine, header)
    output.set_qform(original.get_qform(), int(original.header["qform_code"]))
    output.set_sform(original.get_sform(), int(original.header["sform_code"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, destination)
    saved = nib.load(destination)
    if saved.shape != array.shape or not np.array_equal(np.asanyarray(saved.dataobj), array):
        raise ValueError("Extracted NIfTI changed voxel values")
    for form in ("qform", "sform"):
        if int(saved.header[form + "_code"]) != int(original.header[form + "_code"]):
            raise ValueError("Extracted NIfTI changed spatial form code")
        if not np.allclose(
            getattr(saved, "get_" + form)(), getattr(original, "get_" + form)(), atol=1e-5, rtol=0
        ):
            raise ValueError("Extracted NIfTI changed spatial form")


def stage_case(row, raw):
    destinations = [
        raw / "imagesTr" / (row["case_id"] + "_0000.nii.gz"),
        raw / "labelsTr" / (row["case_id"] + ".nii.gz"),
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Staging refuses existing extracted case files")
    for key in ("image", "label"):
        if digest(row[key]) != row[key + "_sha256"]:
            raise ValueError("Original input checksum differs from frozen manifest")
    image, label = nib.load(row["image"]), nib.load(row["label"])
    if len(image.shape) != 4 or image.shape[-1] != 4 or image.shape[:3] != label.shape:
        raise ValueError("Expected paired four-channel MRI and 3D mask")
    if not np.allclose(image.affine, label.affine, atol=1e-5, rtol=0):
        raise ValueError("Image and label grids differ")
    t1 = np.asanyarray(image.dataobj[..., 1])
    mask = np.asanyarray(label.dataobj)
    if not np.isfinite(t1).all() or not set(np.unique(mask).tolist()) <= {0, 1, 2, 3}:
        raise ValueError("Nonfinite T1 or invalid original label values")
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        raise ValueError("Whole-tumor mask is empty")
    spatial_copy(t1, image, destinations[0])
    spatial_copy(binary, label, destinations[1])
    return {str(path.relative_to(raw.parent.parent)): digest(path) for path in destinations}


def stage(rows, workspace, guard, receipt):
    raw, pre = workspace / "raw" / DATASET, workspace / "preprocessed" / DATASET
    if raw.exists() or pre.exists():
        raise FileExistsError("Staging requires a fresh dataset workspace")
    files = receipt["files"] = {}
    for row in rows:
        guard.check()
        files.update(stage_case(row, raw))
        guard.check()
        receipt["cases_completed"] = len(files) // 2
        print(f"Staged {receipt['cases_completed']}/{len(rows)} T1 cases", flush=True)
    dataset = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "whole_tumor": 1},
        "numTraining": len(rows),
        "file_ending": ".nii.gz",
        # Avoid native utility's separate, default 12-process filename-discovery pool.
        "dataset": {
            r["case_id"]: {
                "images": [f"imagesTr/{r['case_id']}_0000.nii.gz"],
                "label": f"labelsTr/{r['case_id']}.nii.gz",
            }
            for r in rows
        },
    }
    splits = [
        {role: [r["case_id"] for r in rows if r["role"] == role] for role in ("train", "val")}
    ]
    for path, value in [
        (raw / "dataset.json", dataset),
        (pre / "dataset.json", dataset),
        (pre / "splits_final.json", splits),
    ]:
        save_json(path, value)
        files[str(path.relative_to(workspace))] = digest(path)
    (workspace / "results").mkdir(exist_ok=True)


def verify_phase(workspace, phase, manifest_sha):
    receipt = read_json(workspace / "preparation" / (phase + ".json"))
    if receipt.get("status") != "COMPLETE" or receipt.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"Completed matching {phase} receipt required")
    for relative, expected in receipt.get("files", {}).items():
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace) or digest(path) != expected:
            raise ValueError(f"Preserved {phase} file checksum mismatch: {relative}")
    return receipt


def configure_native(workspace):
    if importlib.metadata.version("nnunetv2") != "2.8.1":
        raise RuntimeError("Brain preparation requires pinned nnunetv2==2.8.1")
    for name, folder in [("raw", "raw"), ("preprocessed", "preprocessed"), ("results", "results")]:
        os.environ["nnUNet_" + name] = str(workspace / folder)


def plan(rows, workspace, guard, receipt):
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import (
        DatasetFingerprintExtractor,
    )
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import (
        ExperimentPlanner,
    )

    guard.check()
    train_ids = [r["case_id"] for r in rows if r["role"] == "train"]
    extractor = DatasetFingerprintExtractor(802, num_processes=1, verbose=False)
    extractor.dataset = {case: extractor.dataset[case] for case in train_ids}
    extractor.dataset_json = copy.deepcopy(extractor.dataset_json)
    extractor.dataset_json["numTraining"] = len(train_ids)
    extractor.dataset_json["dataset"] = copy.deepcopy(extractor.dataset)
    extractor.run(overwrite_existing=False)
    guard.check()
    planner = ExperimentPlanner(802, gpu_memory_target_in_gb=2, plans_name=PLANS)
    planner.dataset = {case: planner.dataset[case] for case in train_ids}
    planner.dataset_json = copy.deepcopy(planner.dataset_json)
    planner.dataset_json["numTraining"] = len(train_ids)
    planner.dataset_json["dataset"] = copy.deepcopy(planner.dataset)
    plans = planner.plan_experiment()
    save_json(workspace / "preparation" / "native-plan.json", plans)
    plans["configurations"] = {"2d": plans["configurations"]["2d"]}
    original_batch = plans["configurations"]["2d"]["batch_size"]
    plans["configurations"]["2d"]["batch_size"] = min(10, original_batch)
    target = workspace / "preprocessed" / DATASET / (PLANS + ".json")
    save_json(target, plans)
    guard.check()
    receipt.update(
        training_ids=train_ids,
        source_only_fingerprint=True,
        source_only_planning=True,
        native_batch_size=original_batch,
        final_batch_size=plans["configurations"]["2d"]["batch_size"],
        gpu_memory_target_in_gb=2,
        files={},
    )
    for path in [
        target,
        target.with_name("dataset_fingerprint.json"),
        workspace / "preparation" / "native-plan.json",
    ]:
        receipt["files"][str(path.relative_to(workspace))] = digest(path)


@contextmanager
def compression_threads():
    """Cap native Blosc2's explicit four-thread calls; preserve its numerical code."""
    import blosc2

    original = blosc2.asarray

    def bounded(*args, **kwargs):
        for key in ("cparams", "dparams"):
            params = dict(kwargs.get(key) or {})
            params["nthreads"] = min(2, params.get("nthreads", 2))
            kwargs[key] = params
        return original(*args, **kwargs)

    blosc2.asarray = bounded
    try:
        yield
    finally:
        blosc2.asarray = original


def cache_paths(workspace, cache, case):
    return [cache / (case + suffix) for suffix in (".b2nd", "_seg.b2nd", ".pkl")] + [
        workspace / "preprocessed" / DATASET / "gt_segmentations" / (case + ".nii.gz")
    ]


def preprocess(rows, workspace, guard, receipt, pilot, prior_pilot=None):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    pre = workspace / "preprocessed" / DATASET
    raw = workspace / "raw" / DATASET
    manager = PlansManager(str(pre / (PLANS + ".json")))
    configuration = manager.get_configuration("2d")
    cache = pre / configuration.data_identifier
    cache.mkdir(exist_ok=True)
    (pre / "gt_segmentations").mkdir(exist_ok=True)
    processor = configuration.preprocessor_class(verbose=False)
    selected = [r for r in rows if r["role"] == "train"][:8] if pilot else rows
    prior_files = (prior_pilot or {}).get("files", {})
    actual_files = {str(p.relative_to(workspace)) for p in cache.iterdir() if p.is_file()}
    actual_files |= {str(p.relative_to(workspace)) for p in (pre / "gt_segmentations").iterdir()}
    if actual_files != set(prior_files):
        raise FileExistsError("Cache contains files not covered by the verified pilot receipt")
    files = receipt["files"] = dict(prior_files)
    sizes = []
    with compression_threads():
        for row in selected:
            guard.check()
            case = row["case_id"]
            paths = cache_paths(workspace, cache, case)
            if all(str(path.relative_to(workspace)) in prior_files for path in paths):
                continue
            if any(path.exists() for path in paths):
                raise FileExistsError("Preprocessing refuses unverified existing case outputs")
            processor.run_case_save(
                str(cache / case),
                [str(raw / "imagesTr" / (case + "_0000.nii.gz"))],
                str(raw / "labelsTr" / (case + ".nii.gz")),
                manager,
                configuration,
                read_json(raw / "dataset.json"),
            )
            shutil.copy2(raw / "labelsTr" / (case + ".nii.gz"), paths[-1])
            files.update({str(path.relative_to(workspace)): digest(path) for path in paths})
            sizes.append(sum(path.stat().st_size for path in paths))
            guard.check()
            receipt["cases_completed"] = len(files) // 4
            print(f"Preprocessed {receipt['cases_completed']}/{len(rows)} cases", flush=True)
    receipt["compression_nthreads_max"] = 2
    if pilot:
        free = guard.check()
        projected_cache = int(max(max(sizes), 1.5 * np.mean(sizes)) * len(rows))
        additional = projected_cache - sum(sizes) + MODEL_AND_EVALUATION_RESERVE
        receipt.update(
            pilot_ids=[r["case_id"] for r in selected],
            cache_case_bytes=sizes,
            projected_full_cache_bytes=projected_cache,
            model_and_evaluation_reserve_bytes=MODEL_AND_EVALUATION_RESERVE,
            disk_free_after_pilot=free,
            projected_final_free_bytes=free - additional,
            storage_projection_pass=free - additional >= guard.minimum_free,
            projection_scope="Conservative eight-source-case estimate, not a guarantee",
        )
        if not receipt["storage_projection_pass"]:
            raise OSError(
                "Pilot projects insufficient disk; preserve outputs and revise plan explicitly"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "plan", "pilot", "preprocess"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("data/nnunet-brain"))
    parser.add_argument("--stop-file", type=Path, action="append", default=[])
    parser.add_argument("--deadline-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    if args.deadline_seconds < 1:
        parser.error("A positive finite deadline is required")
    workspace = args.workspace.resolve()
    rows = read_manifest(args.manifest)
    manifest_sha = digest(args.manifest)
    receipt_path = workspace / "preparation" / (args.phase + ".json")
    if receipt_path.exists():
        raise FileExistsError("Preserve prior phase receipts; no automatic retry")
    guard = Guard(workspace, args.deadline_seconds, args.stop_file)
    started = time.monotonic()
    receipt = {
        "status": "INCOMPLETE",
        "phase": args.phase,
        "manifest_sha256": manifest_sha,
        "code_sha256": digest(Path(__file__)),
        "cases_completed": 0,
        "minimum_free_bytes": MIN_FREE,
        "deadline_seconds": args.deadline_seconds,
    }

    def stopped(signum, frame):
        raise InterruptedError(f"Preparation signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stopped)
    signal.alarm(args.deadline_seconds)
    try:
        guard.check()
        save_json(receipt_path, receipt)
        if args.phase == "stage":
            stage(rows, workspace, guard, receipt)
        else:
            verify_phase(workspace, "stage", manifest_sha)
            configure_native(workspace)
            if args.phase == "plan":
                plan(rows, workspace, guard, receipt)
            else:
                verify_phase(workspace, "plan", manifest_sha)
                prior = None
                if args.phase == "preprocess":
                    prior = verify_phase(workspace, "pilot", manifest_sha)
                    if not prior.get("storage_projection_pass"):
                        raise ValueError("Pilot storage clearance required")
                preprocess(rows, workspace, guard, receipt, args.phase == "pilot", prior)
        receipt["status"] = "COMPLETE"
    except BaseException as error:
        receipt.update(status="FAILED", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        receipt["elapsed_seconds"] = time.monotonic() - started
        save_json(receipt_path, receipt)


if __name__ == "__main__":
    main()
