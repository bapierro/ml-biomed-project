"""Run UniverSeg on a whole MRI with a balanced support pool from separate patients."""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from ml_biomed_project.devices import select_device


def preprocess_volume(volume, *, mask=False):
    """Canonical RAS, volume-level MRI normalization, and a 128-cubed grid."""
    if len(volume.shape) != 3 or not all(volume.shape):
        raise ValueError("Expected a nonempty 3D volume.")
    if not np.all(np.isfinite(volume.affine)) or np.isclose(
        np.linalg.det(volume.affine[:3, :3]), 0
    ):
        raise ValueError("Volume affine must be finite and nonsingular.")
    canonical = nib.as_closest_canonical(volume)
    values = canonical.get_fdata(dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("Volume contains nonfinite values.")
    if mask:
        if not np.all((values == 0) | (values == 1)):
            raise ValueError("Support mask must contain only binary 0/1 values.")
    else:
        nonzero = values[values != 0]
        if nonzero.size:
            low, high = np.percentile(nonzero, [0.5, 99.5])
            values = np.clip(values, low, high)
            low, high = float(values.min()), float(values.max())
            values = (values - low) / (high - low) if high > low else np.zeros_like(values)
        else:
            values = np.zeros_like(values)
    tensor = torch.from_numpy(values.astype(np.float32, copy=True))[None, None]
    if mask:
        # Match voxel-center sampling; nearest-exact preserves binary class labels.
        return F.interpolate(tensor, size=(128, 128, 128), mode="nearest-exact")[0, 0]
    return F.interpolate(tensor, size=(128, 128, 128), mode="trilinear", align_corners=False)[0, 0]


def build_support(image_paths, mask_paths, count, query_path):
    """Select exactly count distinct processed slices, balanced across support patients."""
    if not image_paths or len(image_paths) != len(mask_paths):
        raise ValueError("Provide one mask for each support image.")
    if count < len(image_paths):
        raise ValueError("Support count must allow at least one slice per patient.")
    patient_ids = [p.name.removesuffix("_0000.nii.gz") for p in image_paths]
    query_id = query_path.name.removesuffix("_0000.nii.gz")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Duplicate support patient identifiers.")
    if query_id in patient_ids or query_path.resolve() in [p.resolve() for p in image_paths]:
        raise ValueError("The query must not appear in the support pool.")
    images, labels, records = [], [], []
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths, strict=True)):
        image, mask = nib.load(image_path), nib.load(mask_path)
        if image.shape != mask.shape or not np.allclose(
            image.affine, mask.affine, rtol=0, atol=1e-5
        ):
            raise ValueError(f"Support image/mask grid mismatch: {image_path}")
        if image.header.get_xyzt_units()[0] != mask.header.get_xyzt_units()[0]:
            raise ValueError(f"Support spatial units mismatch: {image_path}")
        data, annotation = preprocess_volume(image), preprocess_volume(mask, mask=True)
        candidates = np.flatnonzero(np.any(annotation.numpy(), axis=(0, 1)))
        quota = count // len(image_paths) + (i < count % len(image_paths))
        if len(candidates) < quota:
            raise ValueError(
                f"{image_path.name} has {len(candidates)} positive processed slices; "
                f"{quota} required. No silent reduction or repeated slices."
            )
        selected = candidates[np.linspace(0, len(candidates) - 1, quota, dtype=int)]
        native_depth = nib.as_closest_canonical(image).shape[2]
        for z in selected:
            images.append(data[:, :, z][None])
            labels.append(annotation[:, :, z][None])
            records.append(
                {
                    "patient": patient_ids[i],
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "z_128": int(z),
                    "native_z_approx": float((z + 0.5) * native_depth / 128 - 0.5),
                    "foreground_pixels": int(annotation[:, :, z].sum()),
                }
            )
    return torch.stack(images)[None], torch.stack(labels)[None], records


def restore_prediction(probabilities, original):
    """Restore probabilities to the original grid before thresholding at 0.5."""
    canonical = nib.as_closest_canonical(original)
    restored = F.interpolate(
        probabilities[None, None], size=canonical.shape, mode="trilinear", align_corners=False
    )[0, 0].numpy()
    inverse = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(canonical.affine),
        nib.orientations.io_orientation(original.affine),
    )
    return nib.orientations.apply_orientation(restored >= 0.5, inverse).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--support-image", type=Path, nargs="+", required=True)
    parser.add_argument("--support-mask", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--support-slices", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve earlier results.")

    started = time.perf_counter()
    original = nib.load(args.image)
    query = preprocess_volume(original)
    support_images, support_labels, support_records = build_support(
        args.support_image, args.support_mask, args.support_slices, args.image
    )
    if args.support_slices != support_images.shape[1]:
        raise ValueError("Actual support count differs from the requested count.")

    from universeg import universeg

    device = select_device(args.device)
    model = universeg(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    support_images, support_labels = support_images.to(device), support_labels.to(device)
    probabilities = torch.zeros((128, 128, 128), dtype=torch.float32)
    with torch.inference_mode():
        for z in range(128):
            target = query[:, :, z][None, None].to(device)
            probabilities[:, :, z] = (
                model(target, support_images, support_labels).sigmoid()[0, 0].cpu()
            )
            if z == 0 or (z + 1) % 16 == 0:
                print(f"Slice {z + 1}/128; support={support_images.shape[1]}", flush=True)

    prediction = restore_prediction(probabilities, original)
    if prediction.shape != original.shape:
        raise ValueError("Restored prediction shape does not match the original image.")
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(prediction, original.affine, header), args.output)
    details = {
        "query": str(args.image),
        "support_size": len(support_records),
        "support_patients": sorted({r["patient"] for r in support_records}),
        "support_slices": support_records,
        "device": str(device),
        "threshold": 0.5,
        "preprocessing": "Nonzero MRI percentiles 0.5/99.5, volume min-max, canonical RAS 128^3",
        "ensemble_size": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "foreground_voxels": int(prediction.sum()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps({k: v for k, v in details.items() if k != "support_slices"}, indent=2))


if __name__ == "__main__":
    main()
