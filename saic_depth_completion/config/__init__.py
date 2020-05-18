from saic_depth_completion.utils.registry import Registry
from saic_depth_completion.config.lrn import _C as lrn_cfg
from saic_depth_completion.config.dm_lrn import _C as dm_lrn_cfg

CONFIGS = Registry()

CONFIGS["LRN"] = lrn_cfg
CONFIGS["DM-LRN"] = dm_lrn_cfg


def get_default_config(type):
    return CONFIGS[type].clone()
