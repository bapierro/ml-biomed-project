"""Planar SSC4 (radius1 default) and SSC8 (radius1+2 ablation).

Channels are up-left, up-right, down-left, down-right for radius1, then radius2.
Each block averages squared differences in a reflected3x3 patch BEFORE scaling.
These are in-plane pixel radii, not physical distances or a3D SSC neighborhood.
"""

import torch
import torch.nn.functional as F


def _block(image: torch.Tensor, radius: int) -> torch.Tensor:
    padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    up = padded[:, :, : -2 * radius, radius:-radius]
    down = padded[:, :, 2 * radius :, radius:-radius]
    left = padded[:, :, radius:-radius, : -2 * radius]
    right = padded[:, :, radius:-radius, 2 * radius :]
    distance = torch.cat(
        (
            (up - left).square(),
            (up - right).square(),
            (down - left).square(),
            (down - right).square(),
        ),
        dim=1,
    )
    distance = F.avg_pool2d(F.pad(distance, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    variance = image.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
    scale = torch.maximum(distance.mean(dim=1, keepdim=True), 1e-3 * variance)
    denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.exp(-(distance - distance.amin(dim=1, keepdim=True)) / denominator)


def _validate(image: torch.Tensor, minimum: int, name: str) -> None:
    if image.dtype != torch.float32:
        raise TypeError(f"{name} requires float32 input")
    if image.ndim != 4 or image.shape[0] == 0 or image.shape[1] != 1:
        raise ValueError(f"{name} requires nonempty Bx1xHxW input")
    if min(image.shape[-2:]) < minimum:
        raise ValueError(f"{name} requires H,W>={minimum} for reflected patches")


def ssc4(image: torch.Tensor) -> torch.Tensor:
    """Map finite float32 Bx1xHxW to Bx4xHxW using radius1 PATCH3, H,W>=2."""
    _validate(image, 2, "ssc4")
    return _block(image, 1)


def ssc8(image: torch.Tensor) -> torch.Tensor:
    """Map finite float32 Bx1xHxW input to Bx8xHxW, with H,W>=3.

    Statistics are per image/tile and separately normalized for each four-channel
    block. Constant images produce ones. Nonfinite inputs are not silently repaired.
    No mask, raw-intensity channel, learned parameters or external files are used.
    """
    _validate(image, 3, "ssc8")
    return torch.cat((_block(image, 1), _block(image, 2)), dim=1)
