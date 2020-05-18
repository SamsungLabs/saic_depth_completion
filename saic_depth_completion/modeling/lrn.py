from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


from saic_depth_completion.modeling.backbone import build_backbone
from saic_depth_completion.modeling.blocks import AdaptiveBlock, MaskEncoder, FusionBlock, CRPBlock
from saic_depth_completion.utils import registry
from saic_depth_completion import ops
from saic_depth_completion.metrics import LOSSES



@registry.MODELS.register("LRN")
class LRN(nn.Module):
    def __init__(self, model_cfg):
        super(LRN, self).__init__()

        self.predict_log_depth      = model_cfg.predict_log_depth
        self.losses                 = model_cfg.criterion
        self.activation             = model_cfg.activation
        self.channels               = model_cfg.max_channels
        self.upsample               = model_cfg.upsample
        self.use_crp                = model_cfg.use_crp
        self.input_mask             = model_cfg.input_mask

        in_ch = 4 if not self.input_mask else 5
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=3, kernel_size=7, padding=3),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.backbone = build_backbone(model_cfg.backbone)
        self.feature_channels = self.backbone.feature_channels


        self.fusion_32x16 = FusionBlock(self.channels // 2, self.channels, upsample=self.upsample)
        self.fusion_16x8  = FusionBlock(self.channels // 4, self.channels // 2, upsample=self.upsample)
        self.fusion_8x4   = FusionBlock(self.channels // 8, self.channels // 4, upsample=self.upsample)

        self.adapt1 = nn.Conv2d(self.feature_channels[-1], self.channels, 1, bias=False)
        self.adapt2 = nn.Conv2d(self.feature_channels[-2], self.channels // 2, 1, bias=False)
        self.adapt3 = nn.Conv2d(self.feature_channels[-3], self.channels // 4, 1, bias=False)
        self.adapt4 = nn.Conv2d(self.feature_channels[-4], self.channels // 8, 1, bias=False)

        if self.use_crp:
            self.crp1 = CRPBlock(self.channels, self.channels)
            self.crp2 = CRPBlock(self.channels // 2, self.channels // 2)
            self.crp3 = CRPBlock(self.channels // 4, self.channels // 4)
            self.crp4 = CRPBlock(self.channels // 8, self.channels // 8)


        self.convs = nn.ModuleList([
            nn.Conv2d(self.channels // 8, self.channels // 8, 3, padding=1),
            nn.Conv2d(self.channels // 8, self.channels // 16, 3, padding=1),
            nn.Conv2d(self.channels // 16, self.channels // 16, 3, padding=1),
            nn.Conv2d(self.channels // 16, self.channels // 32, 3, padding=1),
        ])
        self.acts = nn.ModuleList([
            ops.ACTIVATION_LAYERS[self.activation[0]](*self.activation[1]),
            ops.ACTIVATION_LAYERS[self.activation[0]](*self.activation[1]),
            ops.ACTIVATION_LAYERS[self.activation[0]](*self.activation[1]),
            ops.ACTIVATION_LAYERS[self.activation[0]](*self.activation[1]),
        ])

        self.predictor = nn.Conv2d(self.channels // 32, 1, 3, padding=1)

    def criterion(self, pred, gt):
        total = 0
        for spec in self.losses:
            if len(spec) == 3:
                loss_fn = LOSSES[spec[0]](*spec[2])
            else:
                loss_fn = LOSSES[spec[0]]()
            total += spec[1] * loss_fn(pred, gt)
        return total

    def postprocess(self, pred):
        if self.predict_log_depth:
            return pred.exp()
        return pred

    def forward(self, batch):

        color, raw_depth, mask = batch["color"], batch["raw_depth"], batch["mask"]

        if self.input_mask:
            x = torch.cat([color, raw_depth, mask], dim=1)
        else:
            x = torch.cat([color, raw_depth], dim=1)

        x = self.stem(x)

        features = self.backbone(x)[::-1]
        if self.use_crp:
            f1 = self.crp1(self.adapt1(features[0]))
        else:
            f1 = self.adapt1(features[0])
        f2 = self.adapt2(features[1])
        f3 = self.adapt3(features[2])
        f4 = self.adapt4(features[3])

        x = self.fusion_32x16(f2, f1)
        x = self.crp2(x) if self.use_crp else x

        x = self.fusion_16x8(f3, x)
        x = self.crp3(x) if self.use_crp else x

        x = self.fusion_8x4(f4, x)
        x = self.crp4(x) if self.use_crp else x


        x = F.interpolate(x, scale_factor=2, mode=self.upsample)
        x = self.acts[0](self.convs[0](x))
        x = self.acts[1](self.convs[1](x))

        x = F.interpolate(x, scale_factor=2, mode=self.upsample)
        x = self.acts[2](self.convs[2](x))
        x = self.acts[3](self.convs[3](x))

        return self.predictor(x)