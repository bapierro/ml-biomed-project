"""Copy a small portable demo from the completed local study. No downloads/inference.

Readers receiving data/ssc-demo-validation can skip this archive-export step entirely.
Large generated files stay ignored by Git; the original study is never modified.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = "3ff24ee94d4f363348892d80f268fdea6904630a9639606a3b7ccd673ea4b922"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(study, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserve existing bundle: {output}. Choose a fresh --output.")
    training = json.loads((study / "artifacts/training-attempt1.json").read_text())
    if training.get("status") != "COMPLETE" or training.get("native_epochs") != 100:
        raise ValueError("Completed SSC4 source training is required.")
    model = Path(training["output_fold"]).parent
    checkpoint = model / "fold_0/checkpoint_final.pth"
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Expected frozen terminal100 SSC4 checkpoint.")
    original = study.parent / "nnunet-ssc-2026-10-29"
    t2 = study.parent / "t2-validation-2026-11-02"
    evaluation = json.loads((t2 / "spec.json").read_text())
    if json.loads((t2 / "results/comparison.json").read_text()).get("status") != "COMPLETE":
        raise ValueError("Completed frozen SSC4 target evaluation is required.")
    dev_path = original / "metadata/official-development-manifest.json"
    t2_path = Path(evaluation["cohort"]["path"])
    dev = json.loads(dev_path.read_text())
    target = json.loads(t2_path.read_text())
    dev_rows = {r["case_id"]: r for r in dev["rows"] if r["role"] == "val"}
    t2_rows = {r["case_id"]: r for r in target["rows"]}
    source_summary = model / "fold_0/validation/summary.json"
    target_predictions = Path(evaluation["paths"]["predictions"]["SSC4"])
    target_summary = target_predictions / "summary.json"
    summary = json.loads(source_summary.read_text())
    chosen = [
        ("T1", "MCF_T1_0134", "Fixed source illustration; retained prior demo case"),
        ("T1", "EMC014_T1", "Fixed source illustration; retained prior demo case"),
        *[
            ("T2", case, "First case per center, fixed before validation inference")
            for case in json.loads((t2 / "illustrations.json").read_text())
        ],
    ]
    scores = []
    for r in summary["metric_per_case"]:
        case_id = Path(r["prediction_file"]).name.removesuffix(".nii.gz")
        count = r["metrics"]["1"]
        scores.append(
            {
                "case_id": case_id,
                "sequence": "T1",
                "split": "development VAL",
                "patient_group": dev_rows[case_id]["patient_group"],
                "dice": count["Dice"],
                **{k: count[k] for k in ["TP", "FP", "FN", "n_pred", "n_ref"]},
            }
        )
    target_counts = json.loads(target_summary.read_text())
    for r in target_counts["metric_per_case"]:
        case_id = Path(r["prediction_file"]).name.removesuffix(".nii.gz")
        count = r["metrics"]["1"]
        scores.append(
            {
                "case_id": case_id,
                "sequence": "T2",
                "split": "development VAL",
                "patient_group": t2_rows[case_id]["patient_group"],
                "dice": count["Dice"],
                **{k: count[k] for k in ["TP", "FP", "FN", "n_pred", "n_ref"]},
            }
        )
    if (
        len(dev_rows) != 46
        or len(t2_rows) != 56
        or len(scores) != 46 + len(t2_rows)
        or len({(r["sequence"], r["case_id"]) for r in scores}) != 46 + len(t2_rows)
    ):
        raise ValueError("Complete46source/56target score coverage required")
    output.mkdir(parents=True)
    copied = {}

    def copy(source, relative):
        source = Path(source)
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        digest = sha(dest)
        if digest != sha(source):
            raise ValueError(f"Copy mismatch: {source}")
        copied[str(relative)] = {"sha256": digest, "bytes": dest.stat().st_size}
        return str(relative)

    model_files = {}
    for name in ["plans.json", "dataset.json", "fold_0/checkpoint_final.pth"]:
        relative = copy(model / name, Path("model") / name)
        model_files[name] = copied[relative]["sha256"]
    (output / "model/model-metadata.json").write_text(
        json.dumps(
            {
                "files": model_files,
                "checkpoint": "Native terminal100, unchanged bytes",
                "training": "323 T1 TRAIN,46 T1 development VAL,100epochs,seed1729,SSC4-only2D",
            },
            indent=2,
        )
        + "\n"
    )
    cases = []
    for sequence, case_id, selection in chosen:
        row = dev_rows[case_id] if sequence == "T1" else t2_rows[case_id]
        prefix = Path("cases") / case_id
        prediction = (model / "fold_0/validation" if sequence == "T1" else target_predictions) / (
            case_id + ".nii.gz"
        )
        entry = {
            "case_id": case_id,
            "sequence": sequence,
            "selection": selection,
            "split": "development VAL",
            "patient_group": row["patient_group"],
            "image": copy(row["image"]["local_path"], prefix / "image.nii.gz"),
            "reference": copy(row["mask"]["local_path"], prefix / "reference.nii.gz"),
            "predictions": {"SSC4": {"label": 1, "path": copy(prediction, prefix / "ssc4.nii.gz")}},
        }
        cases.append(entry)
    pd.DataFrame(scores).sort_values(["sequence", "case_id"]).to_csv(
        output / "scores.csv", index=False
    )
    # Native checkpoint is already authenticated above. Do not use untrusted pickle checkpoints.
    import torch

    logging = torch.load(checkpoint, map_location="cpu", weights_only=False)["logging"]
    pd.DataFrame(
        {
            "epoch": range(1, 101),
            "train_loss": logging["train_losses"],
            "val_loss": logging["val_losses"],
            "patch_dice": logging["mean_fg_dice"],
            "ema_dice": logging["ema_fg_dice"],
        }
    ).to_csv(output / "training.csv", index=False)
    manifest = {
        "schema_version": 1,
        "cases": cases,
        "files": copied,
        "model": {
            "method": "SSC4",
            "path": "model",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "parameters": 33472684,
            "source_train": 323,
            "source_val": 46,
            "epochs": 100,
            "updates": 25000,
            "seed": 1729,
            "input_features": 4,
            "patch": [256, 320],
            "inplane_spacing_mm": [1.25, 1.25],
        },
        "provenance": {
            "source_summary_sha256": sha(source_summary),
            "target_summary_sha256": sha(target_summary),
            "source_manifest_sha256": sha(dev_path),
            "target_cohort_sha256": sha(t2_path),
            "exporter_sha256": sha(__file__),
            "crosspan_repo": "linkai-peng/CrossPan",
            "crosspan_revision": "0da3821e8af2dfa4c22d56e282372614cb426227",
        },
        "scope": "One SSC4 initialization, source development VAL, T2 development VAL. "
        "Operational filename groups are not verified biological identities. Demo cases selected "
        "for illustration; scores.csv contains ALL46 source and56 target scans. No matched RAW "
        "nnU-Net checkpoint.",
        "data_source": "https://huggingface.co/datasets/linkai-peng/CrossPan",
        "terms": "Consult the CrossPan dataset card and original terms before redistribution/use; "
        "this exporter does not grant a new license. No public bundle URL has been published.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    size = sum(f.stat().st_size for f in output.rglob("*") if f.is_file())
    print(f"Prepared {len(cases)} cases, full102-scan scores, and native model at {output}")
    print(f"{size / 1e6:.1f} MB local copy; zero downloads. Folder is relocatable.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=ROOT / "research/acquisition-shift/nnunet-ssc4-2026-10-31"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/ssc-demo-validation")
    args = parser.parse_args()
    prepare(args.study, args.output)


if __name__ == "__main__":
    main()
