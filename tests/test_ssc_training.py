"""Generated tensors and temporary JSON only; never fit/load a learned model."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ml_biomed_project.ssc import SSCNetwork, ssc4, ssc8
from ml_biomed_project.ssc import train as training


def _workspace(tmp_path):
    name = "Dataset801_Test"
    raw = tmp_path / "raw" / name
    prepared = tmp_path / "preprocessed" / name
    raw.mkdir(parents=True)
    prepared.mkdir(parents=True)
    dataset = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "target": 1},
        "numTraining": 2,
        "file_ending": ".nii.gz",
    }
    for folder in (raw, prepared):
        (folder / "dataset.json").write_text(json.dumps(dataset))
    plans = {
        "dataset_name": name,
        "plans_name": "SSC2D",
        "configurations": {
            "2d": {
                "patch_size": [16, 16],
                "data_identifier": "SSC2D_2d",
                "architecture": {
                    "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                    "arch_kwargs": {},
                    "_kw_requires_import": [],
                },
            }
        },
    }
    (prepared / "SSC2D.json").write_text(json.dumps(plans))
    (prepared / "dataset_fingerprint.json").write_text(json.dumps({"generated_fixture": True}))
    (prepared / "splits_final.json").write_text(
        json.dumps([{"train": ["case-a"], "val": ["case-b"]}])
    )
    cache = prepared / "SSC2D_2d"
    cache.mkdir()
    (cache / "case-a.pkl").write_bytes(b"generated metadata placeholder; never read")
    return ["801", "--workspace", str(tmp_path), "--device", "cpu"], prepared


def _dry_setup(argv, capsys):
    assert training.main([*argv, "--dry-run"]) == 0
    return json.loads(capsys.readouterr().out)


def test_ssc4_is_bitexact_first_block_including_boundaries():
    image = torch.randn(2, 1, 7, 9, generator=torch.Generator().manual_seed(1729))
    assert torch.equal(ssc4(image), ssc8(image)[:, :4])
    assert torch.equal(ssc4(torch.zeros(1, 1, 2, 2)), torch.ones(1, 4, 2, 2))


@pytest.mark.parametrize("channels", [4, 8])
def test_fixed_features_train_only_backbone(channels):
    image = torch.randn(1, 1, 7, 9, requires_grad=True)
    backbone = torch.nn.Conv2d(channels, 1, 1, bias=False)
    network = SSCNetwork(backbone, channels=channels)
    network(image).sum().backward()
    assert image.grad is None
    assert backbone.weight.grad is not None and torch.isfinite(backbone.weight.grad).all()
    assert torch.count_nonzero(backbone.weight.grad) > 0
    assert list(network.state_dict()) == ["backbone.weight"]


def test_native_external_discovery_without_network_construction(monkeypatch):
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

    monkeypatch.setenv("nnUNet_extTrainer", str(Path(training.__file__).parent))
    for channels in (4, 8):
        cls = recursive_find_trainer_class_by_name(f"nnUNetTrainerSSC{channels}")
        assert cls.channels == channels
        assert cls.__name__ == f"nnUNetTrainerSSC{channels}"
    raw = recursive_find_trainer_class_by_name("nnUNetTrainerRAW")
    assert raw.__name__ == "nnUNetTrainerRAW"


def test_raw_builder_preserves_native_single_channel_input(monkeypatch):
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    from ml_biomed_project.ssc.trainer import nnUNetTrainerRAW

    calls = []
    native = torch.nn.Identity()

    def build(plans, configuration, inputs, outputs, deep_supervision):
        calls.append((plans, configuration, inputs, outputs, deep_supervision))
        return native

    monkeypatch.setattr(nnUNetTrainer, "build_network_architecture", staticmethod(build))
    configuration = SimpleNamespace(patch_size=[16, 16])
    network = nnUNetTrainerRAW.build_network_architecture(None, configuration, 1, 2, False)
    image = torch.randn(1, 1, 7, 9, generator=torch.Generator().manual_seed(1729))
    assert network is native and torch.equal(network(image), image)
    assert calls == [(None, configuration, 1, 2, False)]
    with pytest.raises(ValueError, match="one MRI input"):
        nnUNetTrainerRAW.build_network_architecture(None, configuration, 4, 2)
    with pytest.raises(ValueError, match="2D plans"):
        nnUNetTrainerRAW.build_network_architecture(
            None, SimpleNamespace(patch_size=[16, 16, 16]), 1, 2
        )


def test_dry_run_has_no_output_or_subprocess(tmp_path, monkeypatch, capsys):
    argv, _ = _workspace(tmp_path)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run spawned training")

    monkeypatch.setattr(training.subprocess, "run", forbidden)
    setup = _dry_setup(argv, capsys)
    assert setup["scientific"]["channels"] == 4
    assert setup["scientific"]["seed"] == 1729
    assert setup["scientific"]["epochs"] == 100
    assert "nnUNetTrainerSSC4" in setup["command"]
    assert setup["command"][1:3] == ["-c", training.NATIVE_ENTRY]
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


def test_raw_and_ssc50_share_inputs_and_have_distinct_run_identity(tmp_path, capsys):
    argv, _ = _workspace(tmp_path)
    ssc = _dry_setup([*argv, "--epochs", "50"], capsys)
    explicit_ssc = _dry_setup([*argv, "--epochs", "50", "--representation", "ssc"], capsys)
    raw = _dry_setup([*argv, "--epochs", "50", "--representation", "raw"], capsys)
    assert explicit_ssc == ssc
    assert "representation" not in ssc["scientific"]  # Preserve the SSC record schema.
    assert raw["scientific"]["representation"] == "raw"
    assert raw["scientific"]["channels"] == 1
    assert ssc["scientific"]["channels"] == 4
    assert raw["scientific"]["input_hashes"] == ssc["scientific"]["input_hashes"]
    assert raw["environment"] == ssc["environment"]
    assert raw["environment"]["SSC_EPOCHS"] == "50"
    assert raw["fold_dir"] != ssc["fold_dir"]
    assert "nnUNetTrainerRAW" in raw["command"]
    with pytest.raises(ValueError, match="only valid with SSC"):
        training.main([*argv, "--representation", "raw", "--channels", "8", "--dry-run"])


@pytest.mark.parametrize("problem", ["missing", "fingerprint", "overlap", "fold"])
def test_manual_split_is_required_without_random_fallback(tmp_path, problem):
    argv, prepared = _workspace(tmp_path)
    if problem == "missing":
        (prepared / "splits_final.json").unlink()
    elif problem == "fingerprint":
        (prepared / "dataset_fingerprint.json").unlink()
    elif problem == "overlap":
        (prepared / "splits_final.json").write_text(
            json.dumps([{"train": ["case-a"], "val": ["case-a"]}])
        )
    else:
        argv += ["--fold", "1"]
    with pytest.raises((ValueError, FileNotFoundError)):
        training.main([*argv, "--dry-run"])


def test_fresh_and_resume_identity_guards(tmp_path, capsys):
    argv, _ = _workspace(tmp_path)
    setup = _dry_setup(argv, capsys)
    folder = Path(setup["fold_dir"])
    folder.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        training.main([*argv, "--dry-run"])
    with pytest.raises(FileNotFoundError, match="checkpoint_latest"):
        training.main([*argv, "--resume", "--dry-run"])
    (folder / "checkpoint_latest.pth").write_bytes(b"generated placeholder; never deserialized")
    (folder / "run.json").write_text(
        json.dumps({"status": "FAILED", "scientific": setup["scientific"], "attempts": []})
    )
    resumed = _dry_setup([*argv, "--resume"], capsys)
    assert resumed["command"][-1] == "--c"
    with pytest.raises(ValueError, match="scientific"):
        training.main([*argv, "--resume", "--epochs", "101", "--dry-run"])
    (folder / "checkpoint_final.pth").write_bytes(b"generated placeholder")
    (folder / "checkpoint_latest.pth").unlink()
    validation_recovery = _dry_setup([*argv, "--resume"], capsys)
    assert validation_recovery["command"][-1] == "--c"
    (folder / "run.json").write_text(
        json.dumps({"status": "COMPLETE", "scientific": setup["scientific"], "attempts": []})
    )
    with pytest.raises(ValueError, match="already complete"):
        training.main([*argv, "--resume", "--dry-run"])


def test_launcher_delegates_to_native_command_and_records_completion(tmp_path, monkeypatch, capsys):
    argv, _ = _workspace(tmp_path)
    setup = _dry_setup(argv, capsys)
    folder = Path(setup["fold_dir"])
    calls = []

    def fake_native(command, *, env, check):
        assert check is True and env["SSC_EPOCHS"] == "100" and env["SSC_SEED"] == "1729"
        calls.append(command)
        (folder / "checkpoint_final.pth").write_bytes(b"generated placeholder; no model trained")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(training.subprocess, "run", fake_native)
    assert training.main(argv) == 0
    record = json.loads((folder / "run.json").read_text())
    assert calls == [setup["command"]]
    assert record["status"] == "COMPLETE" and record["attempts"][0]["exit_code"] == 0
