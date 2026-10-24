"""Check Dice against known answers, independently of a model or dataset."""

import numpy as np
import pytest

from ml_biomed_project.metrics import dice_score


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], 1.0),
        ([1, 1, 0, 0], [0, 0, 1, 1], 0.0),
        ([1, 1, 0, 0], [1, 0, 1, 0], 0.5),
        ([0, 0, 0, 0], [1, 0, 1, 0], 0.0),
        ([1, 0, 1, 0], [0, 0, 0, 0], 0.0),
        ([True, False], [1.0, 0.0], 1.0),
    ],
    ids=["identical", "disjoint", "partial", "empty-prediction", "empty-reference", "dtypes"],
)
def test_known_dice_scores(prediction, reference, expected):
    assert dice_score(prediction, reference) == pytest.approx(expected)


def test_two_empty_masks_are_undefined():
    assert np.isnan(dice_score(np.zeros((2, 3, 4)), np.zeros((2, 3, 4))))


def test_dice_uses_all_voxels_instead_of_mean_slice_scores():
    reference = np.array([[[1, 0], [0, 0]], [[1, 1], [1, 0]]])
    prediction = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
    assert dice_score(prediction, reference) == pytest.approx(0.4)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="matching shapes"):
        dice_score(np.zeros((2, 3, 4)), np.zeros((3, 2, 4)))


@pytest.mark.parametrize(
    "invalid", [[0, 2], [0, -1], [0, 0.5], [0, np.nan], [0, np.inf], ["0", "1"]]
)
@pytest.mark.parametrize("side", ["prediction", "reference"])
def test_nonbinary_values_are_rejected(invalid, side):
    inputs = {"prediction": [0, 1], "reference": [0, 1]}
    inputs[side] = invalid
    with pytest.raises(ValueError, match="binary"):
        dice_score(**inputs)


@pytest.mark.parametrize("empty", [[], np.empty((2, 0, 3)), 0])
def test_empty_arrays_and_scalars_are_rejected(empty):
    with pytest.raises(ValueError, match="nonempty arrays"):
        dice_score(empty, empty)
