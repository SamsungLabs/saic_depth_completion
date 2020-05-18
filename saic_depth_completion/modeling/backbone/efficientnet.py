import sys
import torch.nn as nn
from efficientnet_pytorch import EfficientNet as _EfficientNet
from efficientnet_pytorch.utils import url_map, get_model_params


from collections import namedtuple

StageSpec = namedtuple("StageSpec", ["num_channels", "stage_stamp"],)

efficientnet_b0 = tuple(StageSpec(num_channels=nc, stage_stamp=ss)
    for (nc, ss) in ((24, 3), (40, 4), (112, 9), (320, 16))
)
efficientnet_b1 = tuple(StageSpec(num_channels=nc, stage_stamp=ss)
    for (nc, ss) in ((24, 5), (40, 8), (112, 16), (320, 23))
)
efficientnet_b2 = tuple(StageSpec(num_channels=nc, stage_stamp=ss)
    for (nc, ss) in ((24, 5), (48, 8), (120, 16), (352, 23))
)
efficientnet_b3 = tuple(StageSpec(num_channels=nc, stage_stamp=ss)
    for (nc, ss) in ((32, 5), (48, 8), (136, 18), (384, 26))
)
efficientnet_b4 = tuple(StageSpec(num_channels=nc, stage_stamp=ss)
    for (nc, ss) in ((32, 6), (56, 10), (160, 22), (448, 32))
)
efficientnet_b5 = tuple(StageSpec(num_channels=nc, stage_stamp=ss)
    for (nc, ss) in ((40, 8), (64, 13), (176, 27), (512, 39))
)
efficientnet_b6 = tuple(StageSpec(num_channels=nc, stage_stamp=ss)
    for (nc, ss) in ((40, 9), (72, 15), (200, 31), (576, 45))
)
efficientnet_b7 = tuple(StageSpec(num_channels=nc, stage_stamp=ss)
    for (nc, ss) in ((48, 11), (80, 18), (224, 38), (640, 55))
)

class EfficientNet(_EfficientNet):
    def __init__(self, model_cfg):

        blocks_args, global_params = get_model_params(model_cfg.arch, dict(image_size=None))
        super().__init__(blocks_args, global_params)

        self.multi_scale_output = model_cfg.multi_scale_output
        self.stage_specs = sys.modules[__name__].__getattribute__(model_cfg.arch.replace("-", "_"))
        self.num_blocks = len(self._blocks)

        del self._fc, self._conv_head, self._bn1, self._avg_pooling, self._dropout

    @property
    def feature_channels(self):
        if self.multi_scale_output:
            return tuple([x.num_channels for x in self.stage_specs])
        return self.stage_specs[-1].num_channels


    def forward(self, x):

        x = self._swish(self._bn0(self._conv_stem(x)))

        block_idx = 0.
        features = []
        for stage in [
            self._blocks[:self.stage_specs[0].stage_stamp],
            self._blocks[self.stage_specs[0].stage_stamp:self.stage_specs[1].stage_stamp],
            self._blocks[self.stage_specs[1].stage_stamp:self.stage_specs[2].stage_stamp],
            self._blocks[self.stage_specs[2].stage_stamp:],
        ]:
            for block in stage:
                x = block(
                    x, self._global_params.drop_connect_rate * block_idx / self.num_blocks
                )
                block_idx += 1.


            features.append(x)

        if self.multi_scale_output:
            return tuple(features)
        return tuple([x])
