"""Download a pinned CrossPan sample; model weights are an explicit optional download."""

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import snapshot_download

from ml_biomed_project.data import load_scan_inventory

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0da3821e8af2dfa4c22d56e282372614cb426227"
WEIGHTS = Path("models/universeg/universeg_v1_nf64_ss64_STA.pt")
WEIGHT_URL = f"https://github.com/JJGO/UniverSeg/releases/download/weights/{WEIGHTS.name}"
WEIGHT_SHA256 = "62c98c48c08bc05f84ed19187844d5eb7e6b7a48d2f690f2033fd8de04543884"
EXAMPLES = load_scan_inventory(ROOT / "data" / "crosspan_examples.csv")


def download_universeg_weights():
    """Verify cached weights, or download and verify before placing the final file."""
    destination = ROOT / WEIGHTS
    if destination.exists():
        with destination.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != WEIGHT_SHA256:
                raise ValueError(f"Checkpoint checksum mismatch; remove {destination} and retry.")
        print(f"Verified existing checkpoint: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as file:
        temporary = Path(file.name)
    try:
        print("Downloading UniverSeg checkpoint (4.5 MiB)...", flush=True)
        with urlopen(WEIGHT_URL, timeout=60) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file)
        with temporary.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != WEIGHT_SHA256:
                raise ValueError("Downloaded checkpoint checksum mismatch; file was not installed.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-universeg-weights",
        action="store_true",
        help="Also download the 4.5 MiB UniverSeg checkpoint.",
    )
    args = parser.parse_args()
    print(
        f"CrossPan: {len(EXAMPLES)} MRI/mask pairs from data/crosspan_examples.csv "
        "(38.4 MiB for the committed selection); existing downloads are reused.",
        flush=True,
    )
    files = [
        path
        for row in EXAMPLES
        for path in (
            f"{row['sequence']}/{row['source_split']}/images/{row['case_id']}_0000.nii.gz",
            f"{row['sequence']}/{row['source_split']}/labels/{row['case_id']}.nii.gz",
        )
    ]
    snapshot_download(
        "linkai-peng/CrossPan",
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=files,
        local_dir=ROOT / "data" / "crosspan",
    )
    missing = [path for path in files if not (ROOT / "data" / "crosspan" / path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Selected file was not downloaded from the pinned release: {missing[0]}"
        )
    if args.with_universeg_weights:
        download_universeg_weights()
    print("Downloads ready. The notebook can inspect the images now.")
    print("For an optional model example, see the MRI models and examples section in README.md.")


if __name__ == "__main__":
    main()
