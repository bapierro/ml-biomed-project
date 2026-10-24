"""Read the small pilot selection and check patient separation across sequences."""

import csv
from pathlib import Path


def load_mri_examples(path):
    """Roles are explicit; release folder names do not determine experimental roles."""
    with Path(path).open(encoding="utf-8", newline="") as file:
        examples = list(csv.DictReader(file))
    required = {"role", "sequence", "source_split", "case_id", "patient_group", "identity_basis"}
    if not examples or any(not required <= row.keys() for row in examples):
        raise ValueError(
            "MRI selection must include roles, source paths, patient groups and evidence."
        )
    seen = set()
    seen_patients = set()
    patient_roles = {}
    for row in examples:
        if any(not row[key] or not row[key].strip() for key in required):
            raise ValueError("MRI selection contains an empty required field.")
        if row["role"] not in {"support", "evaluation"} or row["sequence"] not in {"T1", "T2"}:
            raise ValueError("Expected support/evaluation roles and T1/T2 sequences.")
        if row["source_split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown release split.")
        if row["role"] == "support" and row["source_split"] != "train":
            raise ValueError("Support examples must come from the release training folders.")
        if row["identity_basis"] not in {"filename_subject_code", "upstream_original_code"}:
            raise ValueError("Unknown patient identity evidence; review the selection explicitly.")
        if not all(character.isalnum() or character == "_" for character in row["case_id"]):
            raise ValueError("Case IDs must contain only letters, numbers and underscores.")
        case = (row["sequence"], row["case_id"].casefold())
        if case in seen:
            raise ValueError(f"Duplicate selected scan: {row['case_id']}")
        seen.add(case)
        patient = row["patient_group"].strip().casefold()
        if patient in patient_roles and patient_roles[patient] != row["role"]:
            raise ValueError(
                f"Patient overlap between support and evaluation: {row['patient_group']}"
            )
        patient_roles[patient] = row["role"]
        if (row["sequence"], patient) in seen_patients:
            raise ValueError(
                f"Multiple scans of one patient in the same sequence: {row['patient_group']}"
            )
        seen_patients.add((row["sequence"], patient))
    return examples
