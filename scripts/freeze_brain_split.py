"""Build the portable study's local path manifest from a completed MSD brain audit."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(dataset, audit):
    dataset = dataset.resolve()
    summary_path, cases_path = audit / "summary.json", audit / "cases.jsonl"
    summary = json.loads(summary_path.read_text())
    catalog = dataset / "dataset.json"
    if (
        summary["status"] != "COMPLETE"
        or summary["cases_completed"] != 484
        or summary["catalog_sha256"] != sha(catalog)
        or summary["cases_sha256"] != sha(cases_path)
        or summary["exact_channel_duplicates"] != {"T1": [], "T2": []}
    ):
        raise ValueError("Expected the complete duplicate-free, hash-matched 484-case audit")
    rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
    ids = [r["case_id"] for r in rows]
    catalog_ids = {
        Path(entry["image"]).name.removesuffix(".nii.gz")
        for entry in json.loads(catalog.read_text())["training"]
    }
    if (
        len(set(ids)) != len(ids)
        or len(ids) != 484
        or set(ids) != catalog_ids
        or not all(re.fullmatch(r"BRATS_[0-9]+", case) for case in ids)
    ):
        raise ValueError("Audit case IDs do not match the labeled dataset")
    held_out = set(
        sorted(ids, key=lambda i: (hashlib.sha256(f"brain-t1-t2:1729:{i}".encode()).digest(), i))[
            :97
        ]
    )
    manifest_rows = []
    for row in sorted(rows, key=lambda r: r["case_id"]):
        case = row["case_id"]
        image = dataset / "imagesTr" / f"{case}.nii.gz"
        label = dataset / "labelsTr" / f"{case}.nii.gz"
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(case)
        manifest_rows.append(
            {
                "case_id": case,
                "role": "val" if case in held_out else "train",
                "image": str(image),
                "label": str(label),
                "image_sha256": row["image_sha256"],
                "label_sha256": row["label_sha256"],
            }
        )
    return {
        "schema_version": 1,
        "dataset": "MSD_Task01_BrainTumour",
        "catalog_sha256": summary["catalog_sha256"],
        "audit_sha256": sha(summary_path),
        "split_rule": "SHA256(brain-t1-t2:1729: + case_id), first97 heldout; both modalities retain subject role",
        "channel_indices": {"T1": 1, "T2": 3},
        "binary_foreground_labels": [1, 2, 3],
        "rows": manifest_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved 387 training / 97 paired validation subjects: {args.output}")


if __name__ == "__main__":
    main()
