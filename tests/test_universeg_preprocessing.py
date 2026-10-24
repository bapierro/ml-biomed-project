"""Check preprocessing and support selection without loading a model or medical data."""

import importlib.util
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "run_universeg", Path(__file__).resolve().parents[1] / "scripts" / "run_universeg.py"
)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_volume_normalization_clips_outlier_without_normalizing_slices_separately():
    values = np.full((16, 16, 16), 10, dtype=np.float32)
    values[:, :, 8:] = 100
    values[0, 0, 0] = 1e9
    processed = script.preprocess_volume(nib.Nifti1Image(values, np.eye(4)))
    assert processed.shape == (128, 128, 128)
    assert processed.dtype == torch.float32
    assert torch.isfinite(processed).all()
    assert float(processed.min()) == 0
    assert float(processed.max()) == 1
    assert float(processed[64, 64, 8]) == 0
    assert float(processed[64, 64, 120]) == 1


@pytest.mark.parametrize("value", [0.0, 42.0])
def test_constant_volume_is_finite_zero(value):
    image = nib.Nifti1Image(np.full((4, 4, 4), value, dtype=np.float32), np.eye(4))
    assert torch.count_nonzero(script.preprocess_volume(image)) == 0


def test_image_and_mask_voxel_centers_agree_after_resizing():
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[:8] = 1
    image = nib.Nifti1Image(10 + 100 * mask.astype(np.float32), np.eye(4))
    annotation = nib.Nifti1Image(mask, np.eye(4))
    processed = script.preprocess_volume(image)
    processed_mask = script.preprocess_volume(annotation, mask=True)
    assert set(processed_mask.unique().tolist()) == {0, 1}
    assert torch.equal(processed >= 0.5, processed_mask.bool())


@pytest.mark.parametrize("shape", [(16, 16, 16), (12, 8, 4)])
def test_restore_prediction_preserves_original_shape_and_orientation(shape):
    affine = np.diag([-1.5, -1.5, 3.0, 1.0])
    original = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
    probability = torch.zeros((128, 128, 128))
    probability[:64] = 1
    restored = script.restore_prediction(probability, original)
    assert restored.shape == original.shape
    assert not restored[: shape[0] // 2].any()
    assert restored[shape[0] // 2 :].all()


@pytest.fixture
def support_paths(tmp_path):
    images, masks = [], []
    for i in range(4):
        image_path = tmp_path / f"patient{i}_0000.nii.gz"
        mask_path = tmp_path / f"patient{i}.nii.gz"
        values = np.indices((16, 16, 16))[0].astype(np.float32)
        mask = np.zeros(values.shape, dtype=np.uint8)
        mask[4:12, 4:12] = 1
        nib.save(nib.Nifti1Image(values, np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
        images.append(image_path)
        masks.append(mask_path)
    return images, masks, tmp_path / "query_0000.nii.gz"


def test_support_is_exactly_64_unique_slices_balanced_across_patients(support_paths):
    images, masks, query = support_paths
    support_images, support_labels, records = script.build_support(images, masks, 64, query)
    assert support_images.shape == support_labels.shape == (1, 64, 1, 128, 128)
    assert len({(r["patient"], r["z_128"]) for r in records}) == 64
    assert set(Counter(r["patient"] for r in records).values()) == {16}
    assert (support_labels.sum(dim=(-1, -2, -3)) > 0).all()


def test_insufficient_support_does_not_silently_cap_or_repeat(support_paths):
    images, masks, query = support_paths
    with pytest.raises(ValueError, match="No silent reduction"):
        script.build_support(images[:1], masks[:1], 129, query)


def test_query_cannot_be_used_as_support(support_paths):
    images, masks, _ = support_paths
    with pytest.raises(ValueError, match="query must not appear"):
        script.build_support(images, masks, 64, images[0])


def test_duplicate_patients_are_rejected(support_paths):
    images, masks, query = support_paths
    with pytest.raises(ValueError, match="Duplicate support"):
        script.build_support(images[:1] * 2, masks[:1] * 2, 64, query)
