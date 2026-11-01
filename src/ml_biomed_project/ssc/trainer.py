"""Native external2D trainers: SSC4 default, SSC8 explicit ablation.

    nnUNet_extTrainer=/path/to/ml_biomed_project/ssc

The launcher supplies SSC_EPOCHS/SSC_SEED. Native data augmentation, sampling,
deep supervision, loss, SGD, scheduling and full validation are inherited intact.
"""

import os
import random

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from ml_biomed_project.ssc.network import SSCNetwork


class nnUNetTrainerSSC4(nnUNetTrainer):
    channels = 4

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        epochs = int(os.environ.get("SSC_EPOCHS", "100"))
        seed = int(os.environ.get("SSC_SEED", "1729"))
        if epochs < 1 or not 0 <= seed < 2**32:
            raise ValueError("SSC_EPOCHS must be positive; SSC_SEED must fit uint32")
        self.num_epochs = epochs  # Before initialize creates the native polynomial scheduler.
        self.save_every = 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        if num_input_channels != 1 or len(configuration_manager.patch_size) != 2:
            raise ValueError("SSC trainers require one MRI input and2D plans")
        backbone = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            cls.channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return SSCNetwork(backbone, channels=cls.channels)


class nnUNetTrainerSSC8(nnUNetTrainerSSC4):
    channels = 8
