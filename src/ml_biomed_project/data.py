"""Small scan inventories and patient-group checks, independent of model choice."""

import csv
from pathlib import Path


def load_scan_inventory(path):
    """Read scan facts without assigning experiment roles or imposing a modality."""
    with Path(path).open(encoding="utf-8", newline="") as file:
        scans = list(csv.DictReader(file))
    required = {"sequence", "source_split", "case_id", "patient_group", "identity_basis"}
    if not scans or any(not required <= row.keys() for row in scans):
        raise ValueError("Scan inventory must include source paths, patient groups and evidence.")
    seen = set()
    for row in scans:
        if any(not row[key] or not row[key].strip() for key in required):
            raise ValueError("Scan inventory contains an empty required field.")
        case = (row["sequence"], row["case_id"].casefold())
        if case in seen:
            raise ValueError(f"Duplicate scan: {row['case_id']}")
        seen.add(case)
    return scans


def check_patient_separation(first, second):
    """Reject patient overlap between two sets chosen for an experiment."""
    first_patients = {row["patient_group"].strip().casefold() for row in first}
    second_patients = {row["patient_group"].strip().casefold() for row in second}
    overlap = first_patients & second_patients
    if overlap:
        raise ValueError(f"Patient overlap between selected sets: {', '.join(sorted(overlap))}")
