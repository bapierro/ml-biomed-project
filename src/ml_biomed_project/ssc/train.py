"""Validate a prepared workspace, then delegate training to the native nnU-Net CLI."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ml_biomed_project.devices import select_device

NATIVE_ENTRY = "from nnunetv2.run.run_training import run_training_entry; run_training_entry()"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text())


def _save(path: Path, value: dict):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_setup(args) -> dict:
    """Read JSON/filesystem metadata only. This function never creates or launches anything."""
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    if importlib.metadata.version("nnunetv2") != "2.8.1":
        raise RuntimeError("Training requires nnunetv2==2.8.1")
    if args.epochs < 1 or not 0 <= args.seed < 2**32 or args.fold < 0:
        raise ValueError("Use positive epochs, uint32 seed and a nonnegative integer fold")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.plans):
        raise ValueError("Plans must be a native plans identifier, not a path")
    workspace = args.workspace.expanduser().resolve()
    raw, preprocessed, results = [workspace / name for name in ("raw", "preprocessed", "results")]
    if args.dataset.isdecimal() and 0 <= int(args.dataset) <= 999:
        candidates = sorted(
            p.name for p in preprocessed.glob(f"Dataset{int(args.dataset):03d}_*") if p.is_dir()
        )
        if len(candidates) != 1:
            raise ValueError("Dataset ID must resolve to exactly one prepared dataset")
        dataset_name = candidates[0]
    elif re.fullmatch(r"Dataset[0-9]{3}_[A-Za-z0-9_-]+", args.dataset):
        dataset_name = args.dataset
    else:
        raise ValueError("Dataset must be an integer ID or full DatasetNNN_Name")
    prepared = preprocessed / dataset_name
    paths = {
        "dataset": prepared / "dataset.json",
        "raw_dataset": raw / dataset_name / "dataset.json",
        "plans": prepared / (args.plans + ".json"),
        "fingerprint": prepared / "dataset_fingerprint.json",
        "split": prepared / "splits_final.json",
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(
            "Prepare the raw/preprocessed dataset, plans, fingerprint and manual splits_final.json first"
        )
    dataset = _read(paths["dataset"])
    if dataset != _read(paths["raw_dataset"]):
        raise ValueError("Raw and preprocessed dataset.json differ")
    labels = dataset.get("labels", {})
    if (
        dataset.get("channel_names") != {"0": "MRI"}
        or len(labels) != 2
        or labels.get("background") != 0
        or sorted(labels.values()) != [0, 1]
        or type(dataset.get("numTraining")) is not int
        or dataset["numTraining"] < 2
    ):
        raise ValueError("Expected one MRI channel, background0 and one foreground label1")
    plans = _read(paths["plans"])
    if plans.get("dataset_name") != dataset_name or plans.get("plans_name") != args.plans:
        raise ValueError("Plans dataset/name differ from requested workspace")
    configuration = PlansManager(plans).get_configuration("2d")
    if len(configuration.patch_size) != 2 or configuration.previous_stage_name is not None:
        raise ValueError("Standalone2D plans are required")
    cache = (prepared / configuration.data_identifier).resolve()
    if (
        not cache.is_relative_to(prepared)
        or cache == prepared
        or not cache.is_dir()
        or not any(cache.iterdir())
    ):
        raise FileNotFoundError("Prepare the native2D cache identified by these plans first")
    splits = _read(paths["split"])
    if not isinstance(splits, list) or args.fold >= len(splits):
        raise ValueError(
            "Requested fold is absent from manual splits_final.json; no random fallback"
        )
    for split in splits:
        if not isinstance(split, dict) or not all(
            isinstance(split.get(k), list) and split[k] for k in ("train", "val")
        ):
            raise ValueError("Each manual fold requires nonempty train/val lists")
        train, val = split["train"], split["val"]
        if (
            not all(isinstance(case, str) and case for case in train + val)
            or len(set(train)) != len(train)
            or len(set(val)) != len(val)
            or set(train) & set(val)
            or len(set(train + val)) != dataset["numTraining"]
        ):
            raise ValueError("Manual split IDs must be unique, disjoint and cover numTraining")
    representation = getattr(args, "representation", "ssc")
    if representation not in {"raw", "ssc"} or (representation == "raw" and args.channels != 4):
        raise ValueError("Use raw or ssc; --channels 8 is only valid with SSC")
    channels = 1 if representation == "raw" else args.channels
    trainer = "nnUNetTrainerRAW" if representation == "raw" else f"nnUNetTrainerSSC{channels}"
    folder = results / dataset_name / f"{trainer}__{args.plans}__2d" / f"fold_{args.fold}"
    command = [
        sys.executable,
        "-c",
        NATIVE_ENTRY,
        dataset_name,
        "2d",
        str(args.fold),
        "-tr",
        trainer,
        "-p",
        args.plans,
        "-device",
        str(select_device(args.device)),
    ]
    if args.resume:
        command.append("--c")
    environment = {
        "nnUNet_raw": str(raw),
        "nnUNet_preprocessed": str(preprocessed),
        "nnUNet_results": str(results),
        "nnUNet_extTrainer": str(Path(__file__).resolve().parent),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "2",
        "nnUNet_def_n_proc": "1",
        "nnUNet_n_proc_DA": "1",
        "nnUNet_compile": "false",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "SSC_EPOCHS": str(args.epochs),
        "SSC_SEED": str(args.seed),
    }
    versions = {
        name: importlib.metadata.version(name)
        for name in ("nnunetv2", "torch", "dynamic_network_architectures", "numpy")
    }
    source_files = {
        name: _sha(Path(__file__).with_name(name))
        for name in ("descriptor.py", "network.py", "trainer.py")
    }
    scientific = {
        "dataset": dataset_name,
        "configuration": "2d",
        "plans": args.plans,
        "fold": args.fold,
        "channels": channels,
        "epochs": args.epochs,
        "seed": args.seed,
        "versions": versions,
        "input_hashes": {name: _sha(path) for name, path in paths.items()},
        "source_files": source_files,
    }
    if representation == "raw":
        scientific["representation"] = "raw"
    # SSC retains its existing record schema; trainer names/channels already identify it.
    return {
        "command": command,
        "environment": environment,
        "fold_dir": str(folder),
        "scientific": scientific,
        "train_cases": len(splits[args.fold]["train"]),
        "validation_cases": len(splits[args.fold]["val"]),
    }


def _check_run(setup: dict, resume: bool) -> dict | None:
    folder = Path(setup["fold_dir"])
    if not resume:
        if folder.exists():
            raise FileExistsError(
                "Fresh fit refuses an existing fold; choose another workspace/plans or explicit --resume"
            )
        return None
    checkpoint_exists = any(
        (folder / name).is_file() for name in ("checkpoint_final.pth", "checkpoint_latest.pth")
    )
    if not checkpoint_exists or not (folder / "run.json").is_file():
        raise FileNotFoundError(
            "Resume requires native checkpoint_final.pth or checkpoint_latest.pth and matching run.json; no fresh fallback"
        )
    previous = _read(folder / "run.json")
    if previous.get("status") == "COMPLETE" or previous.get("scientific") != setup["scientific"]:
        raise ValueError("Resume run.json scientific settings differ or run is already complete")
    return previous


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset", help="Dataset ID, e.g.801, or full DatasetNNN_Name")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("data/nnunet"),
        help="Prepared raw/, preprocessed/ and results/ folders",
    )
    parser.add_argument("--plans", default="SSC2D", help="Native plans identifier")
    parser.add_argument(
        "--fold", type=int, default=0, help="Fold index in manual splits_final.json"
    )
    parser.add_argument(
        "--representation",
        choices=["raw", "ssc"],
        default="ssc",
        help="Raw MRI comparator or online SSC features; both use the same native 2D recipe",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=[4, 8],
        default=4,
        help="4: radius1 default;8: radii1+2 ablation",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Native training epochs")
    parser.add_argument(
        "--seed", type=int, default=1729, help="Initialization and augmentation RNG seed"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="auto chooses CUDA, MPS, then CPU",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue matching run.json and native final/latest checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print setup without creating outputs or training",
    )
    args = parser.parse_args(argv)
    setup = build_setup(args)
    _check_run(setup, args.resume)
    if args.dry_run:
        print(json.dumps({"dry_run": True, **setup}, indent=2))
        return 0
    from filelock import FileLock

    folder = Path(setup["fold_dir"])
    folder.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder.parent / f".{folder.name}.ssc.lock"), timeout=0):
        previous = _check_run(setup, args.resume)  # Recheck after the exclusive launcher lock.
        folder.mkdir(exist_ok=True)
        attempts = list(previous.get("attempts", [])) if previous else []
        attempt = {
            "status": "RUNNING",
            "resume": args.resume,
            "started_unix": time.time(),
            "command": setup["command"],
            "environment": setup["environment"],
            "launcher_sha256": _sha(Path(__file__)),
        }
        attempts.append(attempt)
        record = {"status": "RUNNING", "scientific": setup["scientific"], "attempts": attempts}
        path = folder / "run.json"
        _save(path, record)
        started = time.monotonic()
        try:
            subprocess.run(setup["command"], env={**os.environ, **setup["environment"]}, check=True)
            if not (folder / "checkpoint_final.pth").is_file():
                raise RuntimeError("Native command exited without checkpoint_final.pth")
            attempt.update(status="COMPLETE", exit_code=0)
            record["status"] = "COMPLETE"
        except BaseException as error:
            attempt.update(
                status="FAILED",
                exit_code=getattr(error, "returncode", None),
                error=f"{type(error).__name__}: {error}",
            )
            record["status"] = "FAILED"
            raise
        finally:
            attempt.update(elapsed_seconds=time.monotonic() - started, ended_unix=time.time())
            _save(path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
