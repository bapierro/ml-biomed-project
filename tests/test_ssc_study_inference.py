"""Generated metadata/class fixtures only: no learned weights or MRI volumes."""

import copy
import hashlib
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from ml_biomed_project.ssc import inference, study_inference


def bundle(tmp_path, representation="raw"):
    plans = {"image_reader_writer": "SimpleITKIO", "configurations": {"2d": {}}}
    dataset = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "whole_tumor": 1},
        "file_ending": ".nii.gz",
    }
    (tmp_path / "plans.json").write_text(json.dumps(plans))
    (tmp_path / "dataset.json").write_text(json.dumps(dataset))
    (tmp_path / "fold_0").mkdir()
    raw = b"generated non-pickle fixture"
    (tmp_path / "fold_0/checkpoint_final.pth").write_bytes(raw)
    checkpoint = {
        "trainer_name": "nnUNetTrainerRAW" if representation == "raw" else "nnUNetTrainerSSC4",
        "current_epoch": 50,
        "init_args": {"configuration": "2d", "fold": 0, "plans": plans, "dataset_json": dataset},
        "inference_allowed_mirroring_axes": [0, 1],
        "network_weights": {"generated": "sentinel"},
    }
    return hashlib.sha256(raw).hexdigest(), checkpoint


def fake_load(monkeypatch, checkpoint):
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return copy.deepcopy(checkpoint)

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(inference.importlib.metadata, "version", lambda _: "2.8.1")
    return calls


def test_untrusted_hash_rejected_before_deserialization(tmp_path, monkeypatch):
    bundle(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Untrusted pickle was read")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA256"):
        study_inference.load_study_model(tmp_path, "0" * 64, device="cpu")


@pytest.mark.parametrize("representation,channels", [("raw", None), ("ssc4", 4)])
def test_trusted_bundle_has_exact_identity_and_descriptor(
    tmp_path, monkeypatch, representation, channels
):
    digest, checkpoint = bundle(tmp_path, representation)
    calls = fake_load(monkeypatch, checkpoint)
    constructed = []

    def build(*args, **kwargs):
        constructed.append((args, kwargs))
        return "predictor"

    monkeypatch.setattr(study_inference, "_build_native_predictor", build)
    model = study_inference.load_study_model(tmp_path, digest, representation, device="cpu")
    assert model.predictor == "predictor" and model.checkpoint_sha256 == digest
    assert calls[0][1] == {"map_location": "cpu", "weights_only": False}
    assert constructed[0][1] == {"ssc_channels": channels}
    assert constructed[0][0][2]["trainer_name"] == checkpoint["trainer_name"]
    assert model.descriptor.startswith("RAW:" if channels is None else "SSC4:")
    assert set(model.file_hashes) == {"plans.json", "dataset.json", "fold_0/checkpoint_final.pth"}


@pytest.mark.parametrize(
    "mutation",
    ["trainer", "epoch", "configuration", "fold", "bool_fold", "plans", "dataset", "axes"],
)
def test_native_identity_mismatch_prevents_predictor(tmp_path, monkeypatch, mutation):
    digest, checkpoint = bundle(tmp_path)
    if mutation == "trainer":
        checkpoint["trainer_name"] = "nnUNetTrainerSSC4"
    elif mutation == "epoch":
        checkpoint["current_epoch"] = 49
    elif mutation == "configuration":
        checkpoint["init_args"]["configuration"] = "3d_fullres"
    elif mutation in {"fold", "bool_fold"}:
        checkpoint["init_args"]["fold"] = 1 if mutation == "fold" else False
    elif mutation == "plans":
        checkpoint["init_args"]["plans"]["unbound_change"] = True
    elif mutation == "dataset":
        checkpoint["init_args"]["dataset_json"]["numTraining"] = 1
    else:
        checkpoint["inference_allowed_mirroring_axes"] = [0]
    fake_load(monkeypatch, checkpoint)
    monkeypatch.setattr(study_inference, "_build_native_predictor", lambda *a, **k: pytest.fail())
    with pytest.raises(ValueError, match="provenance|mirror axes"):
        study_inference.load_study_model(tmp_path, digest, "raw", device="cpu")


@pytest.mark.parametrize(
    "labels",
    [
        {"background": 0, "tumor": [1, 2]},
        {"background": 0, "tumor": 1, "other": 2},
        {"background": 0, "ignore": 1},
        {"background": False, "tumor": True},
    ],
)
def test_only_binary_nonignored_integer_labels_are_allowed(tmp_path, monkeypatch, labels):
    digest, checkpoint = bundle(tmp_path)
    checkpoint["init_args"]["dataset_json"]["labels"] = labels
    (tmp_path / "dataset.json").write_text(json.dumps(checkpoint["init_args"]["dataset_json"]))
    fake_load(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match="binary labels"):
        study_inference.load_study_model(tmp_path, digest, "raw", device="cpu")


def test_optional_metadata_mismatch_prevents_deserialization(tmp_path, monkeypatch):
    digest, checkpoint = bundle(tmp_path)
    (tmp_path / "model-metadata.json").write_text('{"files": {}}')
    calls = fake_load(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match="metadata hashes"):
        study_inference.load_study_model(tmp_path, digest, "raw", device="cpu")
    assert not calls


@pytest.mark.parametrize("mutation", ["trainer", "epoch", "labels"])
def test_released_loader_keeps_its_original_gate(tmp_path, monkeypatch, mutation):
    _, checkpoint = bundle(tmp_path)
    checkpoint["trainer_name"] = "nnUNetTrainerSSC4_2D100"
    checkpoint["current_epoch"] = 100
    checkpoint["init_args"]["dataset_json"]["labels"] = {"background": 0, "pancreas": 1}
    if mutation == "trainer":
        checkpoint["trainer_name"] = "nnUNetTrainerSSC4"
    elif mutation == "epoch":
        checkpoint["current_epoch"] = 50
    else:
        checkpoint["init_args"]["dataset_json"]["labels"] = {"background": 0, "whole_tumor": 1}
    (tmp_path / "dataset.json").write_text(json.dumps(checkpoint["init_args"]["dataset_json"]))
    fake_load(monkeypatch, checkpoint)
    monkeypatch.setattr(inference, "_sha", lambda _: inference.CHECKPOINT_SHA256)
    monkeypatch.setattr(inference, "_build_native_predictor", lambda *a, **k: pytest.fail())
    with pytest.raises(ValueError, match="provenance|pancreas"):
        inference.load_model(tmp_path, device="cpu")


@pytest.mark.parametrize("channels", [None, 4])
def test_shared_predictor_preserves_native_settings_and_strict_weights(monkeypatch, channels):
    calls = {}

    class Network:
        def load_state_dict(self, weights, strict):
            calls["weights"] = (weights, strict)

        def eval(self):
            return self

        def requires_grad_(self, value):
            calls["grad"] = value
            return self

    backbone = Network()
    configuration = SimpleNamespace(patch_size=[192, 160], previous_stage_name=None)

    class Manager:
        def __init__(self, plans):
            pass

        def get_configuration(self, name):
            assert name == "2d"
            return configuration

        def get_label_manager(self, dataset):
            return SimpleNamespace(num_segmentation_heads=2)

    class Trainer:
        @staticmethod
        def build_network_architecture(manager, conf, inputs, outputs, enable_deep_supervision):
            calls["architecture"] = (inputs, outputs, enable_deep_supervision)
            return backbone

    class Predictor:
        def __init__(self, **kwargs):
            calls["settings"] = kwargs

        def manual_initialization(self, *args):
            calls["manual"] = args
            assert os.environ["nnUNet_compile"] == "false"

    for module_name, attribute, value in [
        ("nnunetv2.inference.predict_from_raw_data", "nnUNetPredictor", Predictor),
        ("nnunetv2.training.nnUNetTrainer.nnUNetTrainer", "nnUNetTrainer", Trainer),
        ("nnunetv2.utilities.plans_handling.plans_handler", "PlansManager", Manager),
    ]:
        module = ModuleType(module_name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, module_name, module)

    def wrap(network, channels):
        calls["ssc"] = channels
        assert network is backbone
        return Network()

    monkeypatch.setattr(inference, "SSCNetwork", wrap)
    monkeypatch.setenv("nnUNet_compile", "original")
    checkpoint = {
        "network_weights": {},
        "trainer_name": "fixture",
        "inference_allowed_mirroring_axes": [0, 1],
    }
    inference._build_native_predictor(
        {}, {}, checkpoint, torch.device("cpu"), ssc_channels=channels
    )
    assert calls["architecture"] == (channels or 1, 2, False)
    assert calls.get("ssc") == channels
    assert calls["weights"] == ({}, True) and calls["grad"] is False
    assert calls["settings"] == {
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
        "perform_everything_on_device": False,
        "device": torch.device("cpu"),
        "verbose": False,
        "verbose_preprocessing": False,
        "allow_tqdm": False,
    }
    assert calls["manual"][-1] == (0, 1)
    assert os.environ["nnUNet_compile"] == "original"


def test_prediction_sidecar_uses_model_descriptor(tmp_path, monkeypatch):
    image = tmp_path / "generated-input.nii.gz"
    image.write_bytes(b"generated reader fixture; not MRI")
    properties = {"sitk_stuff": {"spacing": [1, 1, 1]}}

    class Reader:
        def read_images(self, paths):
            return np.zeros((1, 2, 3, 4), dtype=np.float32), properties

        def write_seg(self, segmentation, path, props):
            from pathlib import Path

            Path(path).write_bytes(b"generated writer fixture; not MRI")

    module = ModuleType("nnunetv2.imageio.simpleitk_reader_writer")
    module.SimpleITKIO = Reader
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(inference.importlib.metadata, "version", lambda _: "fixture")
    predictor = SimpleNamespace(
        predict_single_npy_array=lambda *a, **k: np.zeros((2, 3, 4), dtype=np.uint8)
    )
    model = inference.SSCModel(
        predictor, tmp_path, "0" * 64, torch.device("cpu"), {}, "RAW: fixture"
    )
    output = inference.predict_volume(model, image, tmp_path / "prediction.nii.gz")
    sidecar = json.loads(output.with_name(output.name + ".json").read_text())
    assert sidecar["descriptor"] == "RAW: fixture"
