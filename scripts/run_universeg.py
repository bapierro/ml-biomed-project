"""Run pretrained UniverSeg over a whole MRI using a separate labeled support volume."""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from universeg import universeg

from ml_biomed_project.devices import select_device


def resized_slice(values, *, mask=False):
    """Use the official 128x128 input size; normalize each intensity slice to [0, 1]."""
    values = values.astype(np.float32)
    if not mask:
        low, high = float(values.min()), float(values.max())
        values = (values - low) / (high - low) if high > low else np.zeros_like(values)
    tensor = torch.from_numpy(values.copy())[None, None]
    if mask:
        return F.interpolate(tensor, size=(128, 128), mode="nearest")
    return F.interpolate(tensor, size=(128, 128), mode="bilinear", align_corners=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--support-image", type=Path, required=True)
    parser.add_argument("--support-mask", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--support-slices", type=int, default=4)
    args = parser.parse_args()
    if args.support_slices < 1:
        parser.error("--support-slices must be positive")
    if args.image.resolve() == args.support_image.resolve():
        parser.error("The query and support image must be different patients.")

    started = time.perf_counter()
    original = nib.load(args.image)
    support_original = nib.load(args.support_image)
    support_mask_original = nib.load(args.support_mask)
    if any(
        len(volume.shape) != 3 for volume in (original, support_original, support_mask_original)
    ):
        raise ValueError("Expected 3D query, support image, and support mask.")
    if support_original.shape != support_mask_original.shape or not np.allclose(
        support_original.affine, support_mask_original.affine, rtol=0, atol=1e-5
    ):
        raise ValueError("Support image and mask grids do not match.")

    query = nib.as_closest_canonical(original)
    support = nib.as_closest_canonical(support_original)
    support_mask = nib.as_closest_canonical(support_mask_original)
    query_data = query.get_fdata(dtype=np.float32)
    support_data = support.get_fdata(dtype=np.float32)
    mask_data = np.asarray(support_mask.dataobj)
    if not np.all(np.isfinite(query_data)) or not np.all(np.isfinite(support_data)):
        raise ValueError("Images must contain finite values.")
    if not np.all((mask_data == 0) | (mask_data == 1)):
        raise ValueError("Verify and convert the support annotation to binary 0/1 first.")
    candidates = np.flatnonzero(np.any(mask_data, axis=(0, 1)))
    if not candidates.size:
        raise ValueError("The support mask has no foreground.")
    indices = candidates[
        np.linspace(0, len(candidates) - 1, min(args.support_slices, len(candidates)), dtype=int)
    ]
    support_images = torch.cat([resized_slice(support_data[:, :, z]) for z in indices])[None]
    support_labels = torch.cat([resized_slice(mask_data[:, :, z], mask=True) for z in indices])[
        None
    ]

    device = select_device(args.device)
    model = universeg(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    prediction = np.zeros(query.shape, dtype=np.uint8)
    with torch.inference_mode():
        for z in range(query.shape[2]):
            target = resized_slice(query_data[:, :, z]).to(device)
            probability = model(target, support_images, support_labels).sigmoid().cpu()
            restored = F.interpolate(
                probability, size=query.shape[:2], mode="bilinear", align_corners=False
            )
            prediction[:, :, z] = (restored[0, 0].numpy() >= 0.5).astype(np.uint8)
            print(f"Slice {z + 1}/{query.shape[2]}", flush=True)

    # Canonicalization only permutes/flips voxels. Undo it without spatial interpolation.
    inverse = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(query.affine),
        nib.orientations.io_orientation(original.affine),
    )
    prediction = nib.orientations.apply_orientation(prediction, inverse)
    if prediction.shape != original.shape:
        raise ValueError("Restored prediction shape does not match the original image.")
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(prediction, original.affine, header), args.output)
    details = {
        "query": str(args.image),
        "support_image": str(args.support_image),
        "support_mask": str(args.support_mask),
        "support_slices_canonical": indices.tolist(),
        "device": str(device),
        "threshold": 0.5,
        "elapsed_seconds": time.perf_counter() - started,
        "foreground_voxels": int(prediction.sum()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
