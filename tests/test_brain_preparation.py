"""Generated-volume and routing checks; never read the real brain cohort."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def prepare():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_brain.py"
    spec = importlib.util.spec_from_file_location("prepare_brain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cohort(tmp_path, prepare):
    rows = []
    affine = np.diag([-1.0, 1.5, 2.0, 1.0])
    affine[:3, 3] = [10, 20, -30]
    for i, role in enumerate(["val", "train", "train"], 1):
        case = f"BRATS_{i:03}"
        data = np.stack([np.full((4, 5, 6), i * 10 + c, dtype=np.int16) for c in range(4)], axis=-1)
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[0, 0, :3] = [1, 2, 3]
        row = {"case_id": case, "role": role}
        for name, values in [("image", data), ("label", mask)]:
            path = tmp_path / name / (case + ".nii.gz")
            path.parent.mkdir(exist_ok=True)
            image = nib.Nifti1Image(values, affine)
            image.set_qform(affine, 1)
            image.set_sform(affine, 2)
            nib.save(image, path)
            row[name] = str(path)
            row[name + "_sha256"] = prepare.digest(path)
        rows.append(row)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": rows}))
    return manifest, rows, tmp_path / "workspace"


def test_manifest_requires_unique_frozen_roles_and_hashes(prepare, cohort):
    manifest, rows, _ = cohort
    assert len(prepare.read_manifest(manifest, expected_counts=(2, 1))) == 3
    with pytest.raises(ValueError, match="allocation"):
        prepare.read_manifest(manifest)
    rows[-1]["case_id"] = rows[0]["case_id"]
    manifest.write_text(json.dumps({"rows": rows}))
    with pytest.raises(ValueError, match="unique"):
        prepare.read_manifest(manifest, expected_counts=(2, 1))


def test_stage_extracts_only_t1_and_binary_union_preserving_both_forms(prepare, cohort):
    _, rows, workspace = cohort
    receipt = {}
    guard = prepare.Guard(workspace, 20, minimum_free=0)
    prepare.stage(rows, workspace, guard, receipt)
    raw = workspace / "raw" / prepare.DATASET
    pre = workspace / "preprocessed" / prepare.DATASET
    for i, row in enumerate(rows, 1):
        source = nib.load(row["image"])
        staged = nib.load(raw / "imagesTr" / (row["case_id"] + "_0000.nii.gz"))
        label = nib.load(raw / "labelsTr" / (row["case_id"] + ".nii.gz"))
        assert staged.shape == (4, 5, 6)
        assert np.all(np.asanyarray(staged.dataobj) == i * 10 + 1)
        assert set(np.unique(label.dataobj)) == {0, 1}
        assert np.sum(np.asanyarray(label.dataobj)) == 3
        assert staged.header["qform_code"] == 1
        assert staged.header["sform_code"] == 2
        assert np.array_equal(staged.get_qform(), source.get_qform())
        assert np.array_equal(staged.get_sform(), source.get_sform())
        assert prepare.digest(row["image"]) == row["image_sha256"]
        assert prepare.digest(row["label"]) == row["label_sha256"]
    assert (raw / "dataset.json").read_bytes() == (pre / "dataset.json").read_bytes()
    assert prepare.read_json(pre / "splits_final.json") == [
        {"train": ["BRATS_002", "BRATS_003"], "val": ["BRATS_001"]}
    ]
    assert prepare.read_json(raw / "dataset.json")["numTraining"] == 3
    with pytest.raises(FileExistsError):
        prepare.stage(rows, workspace, guard, {})


def test_corrupt_source_fails_before_extraction(prepare, cohort):
    _, rows, workspace = cohort
    Path(rows[0]["image"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        prepare.stage_case(rows[0], workspace / "raw" / prepare.DATASET)
    assert not workspace.exists()


def test_plan_routes_only_training_ids_and_caps_only_the_2d_batch(prepare, cohort, monkeypatch):
    _, rows, workspace = cohort
    guard = prepare.Guard(workspace, 20, minimum_free=0)
    prepare.stage(rows, workspace, guard, {})
    pre = workspace / "preprocessed" / prepare.DATASET
    dataset = prepare.read_json(pre / "dataset.json")
    calls = []

    class Extractor:
        def __init__(self, dataset_id, num_processes, verbose):
            assert dataset_id == 802 and num_processes == 1
            self.dataset_json = dataset
            self.dataset = dataset["dataset"]

        def run(self, overwrite_existing):
            assert overwrite_existing is False
            calls.append(("fingerprint", sorted(self.dataset), self.dataset_json["numTraining"]))
            prepare.save_json(pre / "dataset_fingerprint.json", {"source": "training only"})

    class Planner:
        def __init__(self, dataset_id, gpu_memory_target_in_gb, plans_name):
            assert (dataset_id, gpu_memory_target_in_gb, plans_name) == (802, 2, "Brain2D")
            self.dataset_json = dataset
            self.dataset = dataset["dataset"]

        def plan_experiment(self):
            calls.append(("plan", sorted(self.dataset), self.dataset_json["numTraining"]))
            return {"configurations": {"2d": {"batch_size": 18}, "3d_fullres": {}}}

    monkeypatch.setitem(
        sys.modules,
        "nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor",
        SimpleNamespace(DatasetFingerprintExtractor=Extractor),
    )
    monkeypatch.setitem(
        sys.modules,
        "nnunetv2.experiment_planning.experiment_planners.default_experiment_planner",
        SimpleNamespace(ExperimentPlanner=Planner),
    )
    receipt = {}
    prepare.plan(rows, workspace, guard, receipt)
    assert calls == [
        ("fingerprint", ["BRATS_002", "BRATS_003"], 2),
        ("plan", ["BRATS_002", "BRATS_003"], 2),
    ]
    assert prepare.read_json(pre / "Brain2D.json")["configurations"] == {"2d": {"batch_size": 10}}
    assert prepare.read_json(pre / "dataset.json")["numTraining"] == 3
    assert (
        "3d_fullres"
        in prepare.read_json(workspace / "preparation/native-plan.json")["configurations"]
    )


def test_compression_wrapper_caps_native_threads_and_restores_function(prepare, monkeypatch):
    calls = []
    original = lambda *a, **kw: calls.append(kw)  # noqa: E731
    fake = SimpleNamespace(asarray=original)
    monkeypatch.setitem(sys.modules, "blosc2", fake)
    params = {"nthreads": 4, "clevel": 5}
    with prepare.compression_threads():
        fake.asarray(np.zeros(2), cparams=params)
    assert fake.asarray is original
    assert calls[0]["cparams"] == {"nthreads": 2, "clevel": 5}
    assert calls[0]["dparams"] == {"nthreads": 2}
    assert params["nthreads"] == 4


def test_receipt_verification_rejects_modified_cache(prepare, tmp_path):
    workspace = tmp_path.resolve()
    cached = workspace / "case.b2nd"
    cached.write_bytes(b"original")
    prepare.save_json(
        workspace / "preparation/pilot.json",
        {
            "status": "COMPLETE",
            "manifest_sha256": "frozen",
            "files": {"case.b2nd": prepare.digest(cached)},
        },
    )
    assert prepare.verify_phase(workspace, "pilot", "frozen")["status"] == "COMPLETE"
    cached.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        prepare.verify_phase(workspace, "pilot", "frozen")


def test_stop_marker_blocks_before_work(prepare, tmp_path):
    (tmp_path / "STOP_REQUESTED").touch()
    with pytest.raises(InterruptedError, match="STOP"):
        prepare.Guard(tmp_path, 20, minimum_free=0).check()


def test_pilot_uses_training_cases_and_full_phase_reuses_only_verified_files(
    prepare, cohort, monkeypatch
):
    _, rows, workspace = cohort
    prepare.stage(rows, workspace, prepare.Guard(workspace, 20, minimum_free=0), {})
    pre = workspace / "preprocessed" / prepare.DATASET
    calls = []

    class Processor:
        def __init__(self, verbose):
            assert verbose is False

        def run_case_save(self, output, images, label, manager, configuration, dataset):
            calls.append(Path(output).name)
            assert "_0000.nii.gz" in images[0]
            assert dataset["numTraining"] == 3
            for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
                Path(output + suffix).write_bytes(b"generated native-cache stand-in")

    configuration = SimpleNamespace(data_identifier="Brain2D_2d", preprocessor_class=Processor)
    manager = SimpleNamespace(get_configuration=lambda name: configuration)
    monkeypatch.setitem(
        sys.modules,
        "nnunetv2.utilities.plans_handling.plans_handler",
        SimpleNamespace(PlansManager=lambda path: manager),
    )
    guard = SimpleNamespace(check=lambda: 30_000_000_000, minimum_free=12_000_000_000)
    pilot = {}
    prepare.preprocess(rows, workspace, guard, pilot, True)
    assert calls == ["BRATS_002", "BRATS_003"]
    assert pilot["pilot_ids"] == calls and pilot["storage_projection_pass"]
    assert len(pilot["files"]) == 8
    pilot.update(status="COMPLETE", manifest_sha256="frozen")
    prepare.save_json(workspace / "preparation/pilot.json", pilot)
    prior = prepare.verify_phase(workspace.resolve(), "pilot", "frozen")
    full = {}
    prepare.preprocess(rows, workspace, guard, full, False, prior)
    assert calls == ["BRATS_002", "BRATS_003", "BRATS_001"]
    assert full["cases_completed"] == 3 and len(full["files"]) == 12
    with pytest.raises(FileExistsError, match="not covered"):
        prepare.preprocess(rows, workspace, guard, {}, False, prior)
    assert (pre / "gt_segmentations/BRATS_001.nii.gz").is_file()


def test_native_blosc_serialization_roundtrip_on_small_generated_arrays(prepare, tmp_path):
    import blosc2
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    data = np.arange(120, dtype=np.float32).reshape(1, 4, 5, 6)
    seg = (data > 60).astype(np.int16)
    prefix = str(tmp_path / "generated")
    with prepare.compression_threads():
        nnUNetDatasetBlosc2.save_case(
            data, seg, {"generated": True}, prefix, chunks=data.shape, blocks=data.shape
        )
    assert np.array_equal(blosc2.open(prefix + ".b2nd")[:], data)
    assert np.array_equal(blosc2.open(prefix + "_seg.b2nd")[:], seg)
