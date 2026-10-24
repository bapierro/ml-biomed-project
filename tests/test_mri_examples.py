"""Protect subject separation even when sequence-specific release folders differ."""

import csv
from pathlib import Path

import pytest

from ml_biomed_project.mri_examples import load_mri_examples

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "mri_examples.csv"


def write_selection(tmp_path, rows):
    path = tmp_path / "selection.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_committed_selection_has_disjoint_patient_groups_across_both_sequences():
    rows = load_mri_examples(MANIFEST)
    support = {row["patient_group"] for row in rows if row["role"] == "support"}
    evaluation = {row["patient_group"] for row in rows if row["role"] == "evaluation"}
    assert len(rows) == 10
    assert len(support) == 8 and len(evaluation) == 2
    assert support.isdisjoint(evaluation)


def test_other_sequence_of_evaluation_patient_cannot_be_support(tmp_path):
    rows = load_mri_examples(MANIFEST)
    # Different scan ID, sequence and release folder; the subject must still be excluded.
    rows[6].update(
        case_id="EMC014_T2", patient_group=" emc:14 ", identity_basis="filename_subject_code"
    )
    with pytest.raises(ValueError, match="Patient overlap"):
        load_mri_examples(write_selection(tmp_path, rows))


def test_two_scans_of_one_patient_cannot_count_as_two_support_patients(tmp_path):
    rows = load_mri_examples(MANIFEST)
    rows[3]["patient_group"] = rows[2]["patient_group"]
    with pytest.raises(ValueError, match="Multiple scans of one patient"):
        load_mri_examples(write_selection(tmp_path, rows))


def test_evaluation_role_is_explicit_even_if_source_folder_is_train(tmp_path):
    rows = load_mri_examples(MANIFEST)
    rows[0]["source_split"] = "train"
    loaded = load_mri_examples(write_selection(tmp_path, rows))
    assert loaded[0]["role"] == "evaluation"


def test_support_does_not_use_release_validation_or_test_folders(tmp_path):
    rows = load_mri_examples(MANIFEST)
    rows[2]["source_split"] = "test"
    with pytest.raises(ValueError, match="release training folders"):
        load_mri_examples(write_selection(tmp_path, rows))
