"""Run the current MRI example; these selections do not define the project's dataset roles."""

import argparse
import os
import shlex
import subprocess
import sys
from importlib.util import find_spec

from download_mri_examples import EXAMPLES, ROOT, WEIGHTS

from ml_biomed_project.data import check_patient_separation

# Choices for this development example only. The data inventory contains no experiment roles.
QUERY_CASES = ("EMC014_T1", "CAD_T2_110")
UNIVERSEG_SUPPORT_CASES = (
    "EMC009_T1",
    "EMC010_T1",
    "EMC011_T1",
    "EMC021_T1",
    "CAD_T2_102",
    "CAD_T2_107",
    "CAD_T2_113",
    "CAD_T2_120",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("universeg", "totalsegmentator"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without running models."
    )
    args = parser.parse_args()
    queries = [row for row in EXAMPLES if row["case_id"] in QUERY_CASES]
    if {row["case_id"] for row in queries} != set(QUERY_CASES):
        parser.error("The current example's query cases are missing from the scan inventory.")
    support_pool = []
    if args.model == "universeg":
        support_pool = [row for row in EXAMPLES if row["case_id"] in UNIVERSEG_SUPPORT_CASES]
        if {row["case_id"] for row in support_pool} != set(UNIVERSEG_SUPPORT_CASES):
            parser.error(
                "The current UniverSeg example's support cases are missing from the inventory."
            )
        check_patient_separation(queries, support_pool)
        patients = [
            (row["sequence"], row["patient_group"].strip().casefold()) for row in support_pool
        ]
        if len(patients) != len(set(patients)):
            parser.error("The current UniverSeg support pool repeats a patient within a sequence.")

    from ml_biomed_project.devices import select_device

    device = select_device(args.device).type
    environment = os.environ.copy()
    environment["TOTALSEG_WEIGHTS_PATH"] = str(ROOT / "models" / "totalsegmentator")
    environment["TOTALSEG_HOME_DIR"] = str(ROOT / "models" / "totalsegmentator-settings")
    for example in queries:
        sequence, split, case = example["sequence"], example["source_split"], example["case_id"]
        image = ROOT / "data" / "crosspan" / sequence / split / "images" / f"{case}_0000.nii.gz"
        required = [image]
        if args.model == "universeg":
            output = (
                ROOT
                / "results"
                / "universeg-preprocessing-64"
                / sequence
                / case
                / "pancreas.nii.gz"
            )
            metadata = output.with_suffix(".json")
            support = [row for row in support_pool if row["sequence"] == sequence]
            images = [
                ROOT
                / "data"
                / "crosspan"
                / sequence
                / row["source_split"]
                / "images"
                / f"{row['case_id']}_0000.nii.gz"
                for row in support
            ]
            masks = [
                ROOT
                / "data"
                / "crosspan"
                / sequence
                / row["source_split"]
                / "labels"
                / f"{row['case_id']}.nii.gz"
                for row in support
            ]
            required += [*images, *masks, ROOT / WEIGHTS]
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_universeg.py"),
                "--image",
                str(image),
                "--support-image",
                *map(str, images),
                "--support-mask",
                *map(str, masks),
                "--weights",
                str(ROOT / WEIGHTS),
                "--output",
                str(output),
                "--support-slices",
                "64",
                "--device",
                device,
            ]
        else:
            output = (
                ROOT
                / "results"
                / "first-mri-smoke-test"
                / "totalsegmentator"
                / case
                / "multilabel.nii.gz"
            )
            metadata = output.with_name("model-report.json")
            command = [
                sys.executable,
                "-m",
                "totalsegmentator.bin.TotalSegmentator",
                "-i",
                str(image),
                "-o",
                str(output),
                "--task",
                "total_mr",
                "--ml",
                "--device",
                "gpu" if device == "cuda" else device,
                "--nr_thr_resamp",
                "1",
                "--nr_thr_saving",
                "1",
                "--report",
                str(metadata),
            ]
        if output.exists():
            if not metadata.exists():
                parser.error(f"Incomplete output: {output}. Move it aside before retrying.")
            print(f"Reusing saved {args.model} prediction: {case}")
            continue
        if metadata.exists():
            parser.error(f"Incomplete output: {metadata}. Move it aside before retrying.")
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        missing = [path for path in required if not path.is_file()]
        if missing:
            weights_option = " --with-universeg-weights" if args.model == "universeg" else ""
            parser.error(
                f"Missing input: {missing[0]}. First run: "
                f"uv run --locked --extra cpu python scripts/download_mri_examples.py{weights_option}"
            )
        if find_spec(args.model) is None:
            parser.error(
                "Install model libraries: uv sync --locked --extra cpu --group models --group notebooks"
            )
        if args.model == "totalsegmentator":
            print("TotalSegmentator may download 442 MiB of MRI weights on first use.", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, cwd=ROOT, env=environment)
    print("Open notebooks/data-visualization-and-universeg.ipynb to inspect the saved outputs.")


if __name__ == "__main__":
    main()
