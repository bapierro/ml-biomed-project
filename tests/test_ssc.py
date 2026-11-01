"""Descriptor contracts independent of a model, dataset, or research-directory import."""

import math

import pytest
import torch

from ml_biomed_project.ssc import ssc8


@pytest.mark.parametrize("value", [0.0, 2.75, -1.0])
def test_constants_are_exactly_one_including_reflected_boundaries(value):
    image = torch.full((2, 1, 7, 9), value, dtype=torch.float32)
    assert torch.equal(ssc8(image), torch.ones((2, 8, 7, 9)))


def test_channel_order_and_independent_radius_normalization_on_a_linear_ramp():
    grid = torch.arange(7, dtype=torch.float32)
    image = (grid[:, None] + 2 * grid[None, :])[None, None]
    # Interior squared differences are [1,9,9,1] at radius1 and4times that
    # at radius2. After subtracting minimum and scaling by mean, both yield:
    expected = torch.tensor([1.0, math.exp(-8 / 5), math.exp(-8 / 5), 1.0] * 2)
    torch.testing.assert_close(ssc8(image)[0, :, 3, 3], expected, rtol=0, atol=1e-7)


def test_reflection_and_pooling_boundary_matches_frozen_reference_fixture():
    grid = torch.arange(7, dtype=torch.float32)
    image = (grid[:, None] + 2 * grid[None, :])[None, None]
    # Frozen AC/V CPU corner fixture, captured independently during equivalence
    # verification. Kept inline so public tests have no research-file dependency.
    expected = torch.tensor(
        [
            0.4581824839115143,
            0.3101399540901184,
            0.3101399540901184,
            1.0,
            0.8668779134750366,
            0.27645304799079895,
            1.0,
            1.0,
        ]
    )
    torch.testing.assert_close(ssc8(image)[0, :, 0, 0], expected, rtol=0, atol=1e-7)


def test_statistics_are_per_image_and_input_is_not_mutated():
    generator = torch.Generator().manual_seed(1729)
    image = torch.randn(1, 1, 11, 13, generator=generator)
    original = image.clone()
    companion = 20 * torch.randn(1, 1, 11, 13, generator=generator) + 100
    together = ssc8(torch.cat((image, companion)))
    assert torch.equal(together[:1], ssc8(image))
    assert torch.equal(image, original)
    assert together.shape == (2, 8, 11, 13)
    assert torch.isfinite(together).all()


@pytest.mark.parametrize(("gain", "offset"), [(2.0, 3.0), (-2.5, 1.25)])
def test_affine_intensity_and_polarity_invariance_with_float_roundoff(gain, offset):
    image = torch.randn(2, 1, 11, 13, generator=torch.Generator().manual_seed(1729))
    torch.testing.assert_close(ssc8(image), ssc8(gain * image + offset), rtol=1e-5, atol=2e-5)


def test_invalid_descriptor_shape_and_dtype_are_explicit():
    with pytest.raises(ValueError, match="H,W>=3"):
        ssc8(torch.ones(1, 1, 2, 3))
    with pytest.raises(ValueError, match="Bx1xHxW"):
        ssc8(torch.ones(1, 2, 7, 9))
    with pytest.raises(TypeError, match="float32"):
        ssc8(torch.ones(1, 1, 7, 9, dtype=torch.float64))
