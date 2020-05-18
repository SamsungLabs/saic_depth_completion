from .relative import *
from .absolute import *

from saic_depth_completion.utils.registry import Registry

LOSSES = Registry()

LOSSES["DepthL2Loss"]       = DepthL2Loss
# LOSSES["DepthLogL2Loss"]    = DepthLogL2Loss
LOSSES["LogDepthL1Loss"]    = LogDepthL1Loss
LOSSES["DepthL1Loss"]       = DepthL1Loss
# LOSSES["DepthLogL1Loss"]    = DepthLogL1Loss
LOSSES["SSIM"]              = SSIM
LOSSES["BerHuLoss"]         = BerHuLoss

