"""Shared online SSC frontend; the learned backbone retains its native state keys."""

import torch
from torch import nn

from ml_biomed_project.ssc.descriptor import ssc4, ssc8


class SSCNetwork(nn.Module):
    """Compute fixed descriptors without gradients, then train/predict the backbone.

    channels=4 is radius1 PATCH3; channels=8 adds the independently normalized
    radius2 block. Neither representation includes the raw intensity channel.
    """

    def __init__(self, backbone: nn.Module, channels: int = 4):
        super().__init__()
        if channels not in (4, 8):
            raise ValueError("SSC channels must be4 or8")
        self.backbone = backbone
        self.channels = channels

    @property
    def decoder(self):
        return self.backbone.decoder

    def forward(self, image: torch.Tensor):
        with torch.no_grad():
            features = ssc4(image) if self.channels == 4 else ssc8(image)
        return self.backbone(features)
