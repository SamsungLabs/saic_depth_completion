from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from saic_depth_completion import ops
from saic_depth_completion.modeling.backbone.res_blocks import Bottleneck


class CRPBlock(nn.Module):
    def conv1x1(self, in_planes, out_planes, stride=1, bias=False):
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                         padding=0, bias=bias)

    def __init__(
            self, in_planes, out_planes, n_stages=4
    ):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(
                self, '{}_{}'.format(i + 1, 'crp'),
                self.conv1x1(
                    in_planes if (i == 0) else out_planes,
                    out_planes, stride=1, bias=False
                )
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'crp'))(top)
            x = top + x
        return x

class FusionBlock(nn.Module):
    def __init__(
            self, hidden_dim, small_planes, activation=("ReLU", []), upsample="bilinear",
    ):
        super(FusionBlock, self).__init__()
        self.act      = ops.ACTIVATION_LAYERS[activation[0]](*activation[1])
        self.upsample = upsample
        self.conv1    = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True)
        self.conv2    = nn.Conv2d(small_planes, hidden_dim, 1, bias=True)

    def forward(self, input1, input2):
        x1 = self.conv1(input1)
        x2 = F.interpolate(
            self.conv2(input2), size=x1.size()[-2:], mode=self.upsample, align_corners=True
        )
        return self.act(x1 + x2)


class MaskEncoder(nn.Module):
    def __init__(
            self, out_ch, scale, kernel_size=3, activation=("ReLU", []),
            round=False, upsample="bilinear",
    ):
        super(MaskEncoder, self).__init__()
        self.scale      = scale
        self.upsample   = upsample
        self.round      = round
        self.convs      = nn.ModuleList([
            nn.Conv2d(1, out_ch // 4, kernel_size, padding=(kernel_size-1)//2),
            nn.Conv2d(out_ch // 4, out_ch // 2, kernel_size, padding=(kernel_size-1)//2),
            nn.Conv2d(out_ch // 2, out_ch, kernel_size, padding=(kernel_size-1)//2)
        ])
        self.acts       = nn.ModuleList([
            ops.ACTIVATION_LAYERS[activation[0]](*activation[1]),
            ops.ACTIVATION_LAYERS[activation[0]](*activation[1]),
            ops.ACTIVATION_LAYERS[activation[0]](*activation[1]),
        ])
    def forward(self, mask):

        mask = F.interpolate(
            mask, scale_factor=1./self.scale, mode=self.upsample
        )
        if self.round:
            mask = torch.round(mask).float()

        x = mask
        for conv, act in zip(self.convs, self.acts):
            x = conv(x)
            x = act(x)
        return x

class SharedEncoder(nn.Module):
    def __init__(
            self, out_channels, scales, in_channels=1, kernel_size=3, upsample="bilinear", activation=("ReLU", [])
    ):
        super(SharedEncoder, self).__init__()
        self.scales = scales
        self.upsample = upsample
        self.feature_extractor = nn.Sequential(*[
            nn.Conv2d(in_channels, 32, kernel_size, padding=(kernel_size - 1) // 2),
            ops.ACTIVATION_LAYERS[activation[0]](*activation[1]),
            nn.Conv2d(32, 64, kernel_size, padding=(kernel_size - 1) // 2),
            ops.ACTIVATION_LAYERS[activation[0]](*activation[1])
        ])

        self.predictors = []
        for oup in out_channels:
            self.predictors.append(
                nn.Sequential(*[
                    nn.Conv2d(64, oup, kernel_size=3, padding=0),
                    ops.ACTIVATION_LAYERS[activation[0]](*activation[1])
                ])
            )
        self.predictors = nn.ModuleList(self.predictors)

    def forward(self, x):
        features = self.feature_extractor(x)
        res = []
        for it, scale in enumerate(self.scales):
            features_scaled = F.interpolate(features, scale_factor=1./scale, mode=self.upsample)
            res.append(
                self.predictors[it](features_scaled)
            )
        return tuple(res)


class AdaptiveBlock(nn.Module):
    def __init__(
            self, x_in_ch, x_out_ch, y_ch, modulation="spade", activation=("ReLU", []), upsample='bilinear'
    ):
        super(AdaptiveBlock, self).__init__()

        x_hidden_ch = min(x_in_ch, x_out_ch)
        self.learned_res = x_in_ch != x_out_ch

        if self.learned_res:
            self.residual = nn.Conv2d(x_in_ch, x_out_ch, kernel_size=1, bias=False)

        self.modulation1 = ops.MODULATION_LAYERS[modulation](x_ch=x_in_ch, y_ch=y_ch, upsample=upsample)
        self.act1        = ops.ACTIVATION_LAYERS[activation[0]](*activation[1])
        self.conv1       = nn.Conv2d(x_in_ch, x_hidden_ch, kernel_size=3, padding=1, bias=True)
        self.modulation2 = ops.MODULATION_LAYERS[modulation](x_ch=x_hidden_ch, y_ch=y_ch, upsample=upsample)
        self.act2        = ops.ACTIVATION_LAYERS[activation[0]](*activation[1])
        self.conv2       = nn.Conv2d(x_hidden_ch, x_out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x, skip):
        if self.learned_res:
            res = self.residual(x)
        else:
            res = x

        x = self.modulation1(x, skip)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.modulation2(x, skip)
        x = self.act2(x)
        x = self.conv2(x)

        return x + res