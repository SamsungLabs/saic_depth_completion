from functools import partial

import torch

from .batch_norm import FrozenBatchNorm2d
from .spade import SPADE, SelfSPADE
from .sean import SEAN

from saic_depth_completion.utils.registry import Registry

MODULATION_LAYERS = Registry()
NORM_LAYERS = Registry()
ACTIVATION_LAYERS = Registry()

ACTIVATION_LAYERS["ReLU"] = torch.nn.ReLU
ACTIVATION_LAYERS["LeakyReLU"] = torch.nn.LeakyReLU

MODULATION_LAYERS["SPADE"] = SPADE
MODULATION_LAYERS["SelfSPADE"] = SelfSPADE
MODULATION_LAYERS["SEAN"] = SEAN

NORM_LAYERS["BatchNorm2d"] = torch.nn.BatchNorm2d
NORM_LAYERS["FrozenBatchNorm2d"] = FrozenBatchNorm2d