"""Keep scan facts independent of experiment selection and model choice."""

import csv
from pathlib import Path

import pytest

from ml_biomed_project.data import check_patient_separation, load_scan_inventory

INVENTORY = Path(__file__).resolve().parents[1] / "data" / "crosspan_examples.csv"


def write_inventory(tmp_path, rows):
    path = tmp_path / "inventory.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_inventory_records_scan_facts_without_experiment_roles():
    rows = load_scan_inventory(INVENTORY)
    assert len(rows) == 10
    assert all("role" not in row for row in rows)


def test_inventory_allows_other_modalities_and_repeat_patient_acquisitions(tmp_path):
    rows = load_scan_inventory(INVENTORY)
    rows[1].update(sequence="CT", source_split="development", identity_basis="dataset_subject_id")
    rows[3]["patient_group"] = rows[2]["patient_group"]
    assert load_scan_inventory(write_inventory(tmp_path, rows)) == rows


def test_experiment_patient_separation_rejects_other_sequence_of_same_patient():
    scans = load_scan_inventory(INVENTORY)
    other_scan = dict(scans[0], sequence="T2", case_id="EMC014_T2", patient_group=" emc:14 ")
    with pytest.raises(ValueError, match="Patient overlap"):
        check_patient_separation([scans[0]], [other_scan])


def test_experiment_patient_separation_accepts_distinct_patients():
    scans = load_scan_inventory(INVENTORY)
    check_patient_separation(scans[:2], scans[2:])


def test_duplicate_scan_records_are_rejected(tmp_path):
    rows = load_scan_inventory(INVENTORY)
    with pytest.raises(ValueError, match="Duplicate scan"):
        load_scan_inventory(write_inventory(tmp_path, rows + rows[:1]))
