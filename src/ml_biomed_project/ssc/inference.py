"""Native nnU-Net inference without importing custom research trainers.

Only the trusted released SSC4 checkpoint is accepted. Native loading uses
pickle (weights_only=False), so its SHA256 is verified BEFORE deserialization.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ml_biomed_project.ssc.network import SSCNetwork

CHECKPOINT_SHA256 = "3ff24ee94d4f363348892d80f268fdea6904630a9639606a3b7ccd673ea4b922"


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _scientific_plans(plans: dict) -> dict:
    result = dict(plans)
    if "continue_training" in result and type(result["continue_training"]) is not bool:
        raise ValueError("Native continue_training flag must be boolean")
    result.pop("continue_training", None)
    return result


@dataclass
class SSCModel:
    predictor: Any = field(repr=False)
    model_dir: Path
    checkpoint_sha256: str
    device: torch.device
    file_hashes: dict[str, str] = field(repr=False)


def load_model(model_dir: str | Path, device: str = "auto") -> SSCModel:
    """Load the released final100epoch SSC4 model, using local files only.

    Required: plans.json,dataset.json,fold_0/checkpoint_final.pth. Optional
    model-metadata.json uses {"files": {relative_filename: sha256}}. The pinned
    checkpoint itself binds the complete plans/dataset, even without that receipt.
    auto chooses CUDA, then MPS, then CPU; explicit unavailable devices fail.
    """
    folder = Path(model_dir).expanduser().resolve()
    names = ["plans.json", "dataset.json", "fold_0/checkpoint_final.pth"]
    hashes = {name: _sha(folder / name) for name in names}
    if hashes[names[2]] != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA256 differs from the trusted released SSC4 model")
    metadata = folder / "model-metadata.json"
    if metadata.exists():
        declared = json.loads(metadata.read_text())["files"]
        if any(declared.get(name) != digest for name, digest in hashes.items()):
            raise ValueError("Model bundle metadata hashes differ")
    if importlib.metadata.version("nnunetv2") != "2.8.1":
        raise RuntimeError("This model requires nnunetv2==2.8.1")
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Use auto,cpu,cuda or mps")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS device is unavailable")
    plans = json.loads((folder / names[0]).read_text())
    dataset = json.loads((folder / names[1]).read_text())
    checkpoint = torch.load(folder / names[2], map_location="cpu", weights_only=False)
    initial = checkpoint.get("init_args", {})
    if (
        checkpoint.get("trainer_name") != "nnUNetTrainerSSC4_2D100"
        or checkpoint.get("current_epoch") != 100
        or initial.get("configuration") != "2d"
        or initial.get("fold") != 0
        or _scientific_plans(initial.get("plans", {})) != _scientific_plans(plans)
        or initial.get("dataset_json") != dataset
    ):
        raise ValueError("Checkpoint native model/plans/dataset provenance differs")
    if (
        dataset.get("channel_names") != {"0": "MRI"}
        or dataset.get("labels") != {"background": 0, "pancreas": 1}
        or dataset.get("file_ending") != ".nii.gz"
        or plans.get("image_reader_writer") != "SimpleITKIO"
    ):
        raise ValueError("Expected one MRI input,binary pancreas labels and native SimpleITKIO")
    axes = tuple(checkpoint.get("inference_allowed_mirroring_axes") or ())
    if axes != (0, 1):
        raise ValueError("Released2D mirror axes must be (0,1)")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    manager = PlansManager(plans)
    configuration = manager.get_configuration("2d")
    if len(configuration.patch_size) != 2 or configuration.previous_stage_name is not None:
        raise ValueError("Standalone2D source plans required")
    backbone = nnUNetTrainer.build_network_architecture(
        manager,
        configuration,
        4,
        manager.get_label_manager(dataset).num_segmentation_heads,
        enable_deep_supervision=False,
    )
    network = SSCNetwork(backbone, channels=4)
    weights = checkpoint["network_weights"]
    network.load_state_dict(weights, strict=True)
    network.eval().requires_grad_(False)

    class FinitePredictor(nnUNetPredictor):
        def predict_logits_from_preprocessed_data(self, *args, **kwargs):
            logits = super().predict_logits_from_preprocessed_data(*args, **kwargs)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite native logits; prediction not exported")
            return logits

    predictor = FinitePredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=selected,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    original_compile = os.environ.get("nnUNet_compile")
    os.environ["nnUNet_compile"] = "false"
    try:
        predictor.manual_initialization(
            network, manager, configuration, [weights], dataset, checkpoint["trainer_name"], axes
        )
    finally:
        if original_compile is None:
            os.environ.pop("nnUNet_compile", None)
        else:
            os.environ["nnUNet_compile"] = original_compile
    del checkpoint  # Drop unused optimizer/logger state; keep only frozen inference parameters.
    return SSCModel(predictor, folder, CHECKPOINT_SHA256, selected, hashes)


def predict_volume(model: SSCModel, image_path: str | Path, output_path: str | Path) -> Path:
    """Predict one native3D MRI with the package's preprocessing/resampling/argmax.

    Input uses SimpleITK's native CZYX ordering and properties. No label input,
    manual normalization, canonical flips, threshold or postprocessing is added.
    A .nii.gz prediction plus a small provenance sidecar are written; no overwrite.
    """
    import numpy as np
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    image = Path(image_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    sidecar = output.with_name(output.name + ".json")
    if not output.name.endswith(".nii.gz"):
        raise ValueError("Use a .nii.gz output path")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Prediction or provenance already exists; choose a new output")
    if any(_sha(model.model_dir / name) != digest for name, digest in model.file_hashes.items()):
        raise ValueError("Model bundle changed after loading")
    started = time.monotonic()
    image_hash = _sha(image)
    reader = SimpleITKIO()
    data, properties = reader.read_images([str(image)])
    if data.ndim != 4 or data.shape[0] != 1 or len(properties["sitk_stuff"]["spacing"]) != 3:
        raise ValueError("A single-channel3D MRI volume is required")
    if not np.isfinite(data).all():
        raise ValueError("MRI contains nonfinite intensities")
    segmentation = model.predictor.predict_single_npy_array(
        data,
        properties,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    )
    if segmentation.shape != data.shape[1:] or not np.isin(segmentation, [0, 1]).all():
        raise ValueError("Native binary prediction shape/values are invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    reader.write_seg(segmentation, str(output), properties)
    provenance = {
        "image": str(image),
        "image_sha256": image_hash,
        "output": str(output),
        "output_sha256": _sha(output),
        "model_dir": str(model.model_dir),
        "checkpoint_sha256": model.checkpoint_sha256,
        "model_file_hashes": model.file_hashes,
        "device": str(model.device),
        "elapsed_seconds": time.monotonic() - started,
        "configuration": "2d",
        "descriptor": "SSC4: radius1,reflect3x3 SSDmean; no raw channel",
        "inference": {
            "tile_step_size": 0.5,
            "gaussian": True,
            "mirroring": True,
            "fold": 0,
            "checkpoint": "checkpoint_final.pth",
            "postprocessing": "none",
            "CPU_logit_accumulation": True,
        },
        "versions": {
            "python": platform.python_version(),
            **{
                k: importlib.metadata.version(k)
                for k in [
                    "torch",
                    "nnunetv2",
                    "dynamic_network_architectures",
                    "numpy",
                    "SimpleITK",
                ]
            },
        },
    }
    temporary = sidecar.with_suffix(sidecar.suffix + ".partial")
    temporary.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    temporary.replace(sidecar)
    return output
