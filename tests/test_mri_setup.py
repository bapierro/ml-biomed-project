"""Check fresh-checkout setup without network access or model inference."""

import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest


@pytest.fixture
def scripts(tmp_path, monkeypatch):
    modules = []
    for name in ("download_mri_examples", "run_mri_examples"):
        path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "ROOT", tmp_path)
        modules.append(module)
    return modules


def test_valid_checkpoint_is_reused_without_network(scripts, monkeypatch):
    download, _ = scripts
    path = download.ROOT / download.WEIGHTS
    path.parent.mkdir(parents=True)
    path.write_bytes(b"test checkpoint")
    monkeypatch.setattr(download, "WEIGHT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(download, "urlopen", lambda *a, **kw: pytest.fail("Network was used"))
    download.download_universeg_weights()
    assert path.read_bytes() == b"test checkpoint"


def test_corrupt_checkpoint_download_is_not_installed(scripts, monkeypatch):
    download, _ = scripts
    monkeypatch.setattr(download, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt download"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        download.download_universeg_weights()
    assert not (download.ROOT / download.WEIGHTS).exists()
    assert not list((download.ROOT / download.WEIGHTS).parent.glob("*.part"))


def test_data_download_is_selective_and_pinned(scripts, monkeypatch):
    download, _ = scripts
    calls = []
    monkeypatch.setattr(download, "snapshot_download", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(sys, "argv", ["download_mri_examples.py", "--data-only"])
    download.main()
    assert calls[0][1]["revision"] == download.REVISION
    assert calls[0][1]["repo_type"] == "dataset"
    files = calls[0][1]["allow_patterns"]
    assert len(files) == len(set(files)) == 20
    assert not any("*" in path for path in files)
    assert sum("/val/images/" in path for path in files) == 2
    assert sum("/train/images/" in path for path in files) == 8
    assert not (download.ROOT / download.WEIGHTS).exists()


@pytest.mark.parametrize("model", ["universeg", "totalsegmentator"])
def test_model_runner_uses_the_expected_inputs_and_reuses_completed_outputs(
    scripts, monkeypatch, model
):
    download, runner = scripts
    for sequence, split, case in download.CASES:
        for kind, name in [("images", f"{case}_0000.nii.gz"), ("labels", f"{case}.nii.gz")]:
            path = runner.ROOT / "data" / "crosspan" / sequence / split / kind / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    weights = runner.ROOT / download.WEIGHTS
    weights.parent.mkdir(parents=True)
    weights.touch()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output" if model == "universeg" else "-o") + 1])
        output.touch()
        metadata = (
            output.with_suffix(".json")
            if model == "universeg"
            else output.with_name("model-report.json")
        )
        metadata.write_text("{}")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "find_spec", lambda name: object())
    monkeypatch.setattr(sys, "argv", ["run_mri_examples.py", "--model", model, "--device", "cpu"])
    runner.main()
    assert len(calls) == 2
    for command in calls:
        if model == "universeg":
            images = command[command.index("--support-image") + 1 : command.index("--support-mask")]
            masks = command[command.index("--support-mask") + 1 : command.index("--weights")]
            assert len(images) == len(masks) == 4
            assert all("train" in Path(path).parts for path in images + masks)
            assert command[command.index("--image") + 1] not in images
            assert command[command.index("--support-slices") + 1] == "64"
        else:
            assert command[command.index("--task") + 1] == "total_mr"
            assert "--fast" not in command and "--roi_subset" not in command
            assert "--ml" in command and "--report" in command
    runner.main()
    assert len(calls) == 2


def test_runner_missing_inputs_give_setup_instructions(scripts, monkeypatch, capsys):
    _, runner = scripts
    monkeypatch.setattr(sys, "argv", ["run_mri_examples.py", "--device", "cpu"])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("Inference was run"))
    with pytest.raises(SystemExit, match="2"):
        runner.main()
    assert "download_mri_examples.py" in capsys.readouterr().err


def test_runner_does_not_overwrite_an_incomplete_prediction(scripts, monkeypatch, capsys):
    _, runner = scripts
    output = runner.ROOT / "results/universeg-preprocessing-64/T1/EMC014_T1/pancreas.nii.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"preserve this output")
    monkeypatch.setattr(sys, "argv", ["run_mri_examples.py", "--device", "cpu"])
    with pytest.raises(SystemExit, match="2"):
        runner.main()
    assert "Incomplete output" in capsys.readouterr().err
    assert output.read_bytes() == b"preserve this output"
