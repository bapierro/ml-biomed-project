"""Download four development/support MRI image-mask pairs (about 13 MiB) from CrossPan."""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("T1", "val", "EMC014_T1"),
    ("T2", "val", "CAD_T2_110"),
    ("T1", "train", "EMC009_T1"),
    ("T2", "train", "CAD_T2_102"),
)

if __name__ == "__main__":
    files = [
        path
        for sequence, split, case in CASES
        for path in (
            f"{sequence}/{split}/images/{case}_0000.nii.gz",
            f"{sequence}/{split}/labels/{case}.nii.gz",
        )
    ]
    snapshot_download(
        "linkai-peng/CrossPan",
        repo_type="dataset",
        revision="0da3821e8af2dfa4c22d56e282372614cb426227",
        allow_patterns=files,
        local_dir=ROOT / "data" / "crosspan",
    )
