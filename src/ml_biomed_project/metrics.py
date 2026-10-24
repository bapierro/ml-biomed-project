"""Small evaluation functions shared by notebooks and experiments."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import dice


def dice_score(prediction: ArrayLike, reference: ArrayLike) -> float:
    """Return whole-array binary Dice, or NaN when both masks are empty.

    Masks must have the same nonempty shape and contain only boolean or numeric
    0/1 values. All voxels contribute to one score; slices are not averaged.
    Probabilities and multiclass masks require explicit conversion beforehand.
    Image geometry must be checked separately before passing voxel arrays here.
    """
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must have matching shapes.")
    if prediction.ndim == 0 or prediction.size == 0:
        raise ValueError("Masks must be nonempty arrays, not scalars.")
    for name, mask in (("Prediction", prediction), ("Reference", reference)):
        if mask.dtype.kind not in "buif" or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} must contain only binary boolean or numeric 0/1 values.")

    if not np.any(prediction) and not np.any(reference):
        return float("nan")
    return float(1 - dice(prediction.astype(bool).ravel(), reference.astype(bool).ravel()))
