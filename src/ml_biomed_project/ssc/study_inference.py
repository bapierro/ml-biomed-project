"""Load explicitly hash-pinned local study models without changing the release gate."""

import importlib.metadata
import json
import re
from pathlib import Path

import torch

from ml_biomed_project.ssc.inference import (
    SSCModel,
    _build_native_predictor,
    _scientific_plans,
    _select_device,
    _sha,
)


def load_study_model(
    model_dir: str | Path,
    checkpoint_sha256: str,
    representation: str = "ssc4",
    epochs: int = 50,
    device: str = "auto",
) -> SSCModel:
    """Load one frozen 50-epoch RAW or SSC4 source model, using local files only.

    The caller supplies a SHA256 from its reviewed, frozen study specification;
    do not derive trust by passing the digest of an arbitrary downloaded file.
    That digest is checked BEFORE pickle deserialization. The checkpoint binds
    plans/dataset/fold/trainer/epoch, and weights load strictly. No optimizer is
    constructed or restored, and this function performs no volume inference.

    Required files: plans.json, dataset.json, fold_0/checkpoint_final.pth.
    Optional model-metadata.json uses the same file-hash schema as load_model.
    """
    if representation not in {"raw", "ssc4"}:
        raise ValueError("Study representation must be raw or ssc4")
    if type(epochs) is not int or epochs != 50:
        raise ValueError("This study loader requires the frozen 50-epoch horizon")
    if (
        not isinstance(checkpoint_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None
    ):
        raise ValueError("Supply a trusted lowercase checkpoint SHA256")
    folder = Path(model_dir).expanduser().resolve()
    names = ["plans.json", "dataset.json", "fold_0/checkpoint_final.pth"]
    hashes = {name: _sha(folder / name) for name in names}
    if hashes[names[2]] != checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 differs from the frozen study model")
    metadata = folder / "model-metadata.json"
    if metadata.exists():
        declared = json.loads(metadata.read_text())["files"]
        if any(declared.get(name) != digest for name, digest in hashes.items()):
            raise ValueError("Model bundle metadata hashes differ")
    if importlib.metadata.version("nnunetv2") != "2.8.1":
        raise RuntimeError("This model requires nnunetv2==2.8.1")
    selected = _select_device(device)
    plans = json.loads((folder / names[0]).read_text())
    dataset = json.loads((folder / names[1]).read_text())
    checkpoint = torch.load(folder / names[2], map_location="cpu", weights_only=False)
    initial = checkpoint.get("init_args", {})
    trainer = "nnUNetTrainerRAW" if representation == "raw" else "nnUNetTrainerSSC4"
    if (
        checkpoint.get("trainer_name") != trainer
        or checkpoint.get("current_epoch") != epochs
        or initial.get("configuration") != "2d"
        or type(initial.get("fold")) is not int
        or initial["fold"] != 0
        or _scientific_plans(initial.get("plans", {})) != _scientific_plans(plans)
        or initial.get("dataset_json") != dataset
    ):
        raise ValueError("Checkpoint native model/plans/dataset provenance differs")
    labels = dataset.get("labels", {})
    if (
        dataset.get("channel_names") != {"0": "MRI"}
        or not isinstance(labels, dict)
        or len(labels) != 2
        or labels.get("background") != 0
        or any(not isinstance(name, str) or not name for name in labels)
        or any(type(value) is not int for value in labels.values())
        or sorted(labels.values()) != [0, 1]
        or "ignore" in labels
        or dataset.get("file_ending") != ".nii.gz"
        or plans.get("image_reader_writer") != "SimpleITKIO"
    ):
        raise ValueError("Expected one MRI input, binary labels and native SimpleITKIO")
    if tuple(checkpoint.get("inference_allowed_mirroring_axes") or ()) != (0, 1):
        raise ValueError("Study2D mirror axes must be (0,1)")
    predictor = _build_native_predictor(
        plans, dataset, checkpoint, selected, ssc_channels=None if representation == "raw" else 4
    )
    del checkpoint
    if representation == "raw":
        return SSCModel(
            predictor,
            folder,
            checkpoint_sha256,
            selected,
            hashes,
            descriptor="RAW: native normalized MRI; no SSC frontend",
        )
    return SSCModel(predictor, folder, checkpoint_sha256, selected, hashes)
