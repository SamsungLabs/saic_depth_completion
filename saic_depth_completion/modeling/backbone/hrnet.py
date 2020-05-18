import sys

import torch
from torch import nn
from torch.nn import Conv2d
from collections import namedtuple

from saic_depth_completion.modeling.backbone import res_blocks
from saic_depth_completion import ops

StageSpec = namedtuple(
    "StageSpec",
    [
        "num_channels",         # tuple
        "num_blocks",         # All layers in the same sequence have the same number output channels
        "num_modules",          # Number of residual blocks in the sequence
        "num_branches",      # True => return the last feature map from this sequence
        "block"
    ],
)

hrnet_w18 = tuple(
    StageSpec(num_channels=nc, num_blocks=nbl, num_modules=nm, num_branches=nbr, block=b)
    for (nc, nbl, nm, nbr, b) in (
        ((64),               (4),            1,          1,      "Bottleneck"),
        ((18, 36),           (4, 4),         1,          2,      "BasicBlock"),
        ((18, 36, 72),       (4, 4, 4),      4,          3,      "BasicBlock"),
        ((18, 36, 72, 144),  (4, 4, 4, 4),   3,          4,      "BasicBlock"),
    )
)

hrnet_w18_small_v1 = tuple(
    StageSpec(num_channels=nc, num_blocks=nbl, num_modules=nm, num_branches=nbr, block=b)
    for (nc, nbl, nm, nbr, b) in (
        ((32),               (1),            1,          1,      "Bottleneck"),
        ((16, 32),           (2, 2),         1,          2,      "BasicBlock"),
        ((16, 32, 64),       (2, 2, 2),      1,          3,      "BasicBlock"),
        ((16, 32, 64, 128),  (2, 2, 2, 2),   1,          4,      "BasicBlock"),
    )
)
hrnet_w18_small_v2 = tuple(
    StageSpec(num_channels=nc, num_blocks=nbl, num_modules=nm, num_branches=nbr, block=b)
    for (nc, nbl, nm, nbr, b) in (
        ((64),               (2),            1,          1,      "Bottleneck"),
        ((18, 36),           (2, 2),         1,          2,      "BasicBlock"),
        ((18, 36, 72),       (2, 2, 2),      3,          3,      "BasicBlock"),
        ((18, 36, 72, 144),  (2, 2, 2, 2),   2,          4,      "BasicBlock"),
    )
)

hrnet_w30 = tuple(
    StageSpec(num_channels=nc, num_blocks=nbl, num_modules=nm, num_branches=nbr, block=b)
    for (nc, nbl, nm, nbr, b) in (
        ((64),               (4),            1,          1,      "Bottleneck"),
        ((30, 60),           (4, 4),         1,          2,      "BasicBlock"),
        ((30, 60, 120),      (4, 4, 4),      4,          3,      "BasicBlock"),
        ((30, 60, 120, 240), (4, 4, 4, 4),   3,          4,      "BasicBlock"),
    )
)

hrnet_w32 = tuple(
    StageSpec(num_channels=nc, num_blocks=nbl, num_modules=nm, num_branches=nbr, block=b)
    for (nc, nbl, nm, nbr, b) in (
        ((64),               (4),            1,          1,      "Bottleneck"),
        ((32, 64),           (4, 4),         1,          2,      "BasicBlock"),
        ((32, 64, 128),      (4, 4, 4),      4,          3,      "BasicBlock"),
        ((32, 64, 128, 256), (4, 4, 4, 4),   3,          4,      "BasicBlock"),
    )
)

hrnet_w40 = tuple(
    StageSpec(num_channels=nc, num_blocks=nbl, num_modules=nm, num_branches=nbr, block=b)
    for (nc, nbl, nm, nbr, b) in (
        ((64),               (4),            1,          1,      "Bottleneck"),
        ((40, 80),           (4, 4),         1,          2,      "BasicBlock"),
        ((40, 80, 160),      (4, 4, 4),      4,          3,      "BasicBlock"),
        ((40, 80, 160, 320), (4, 4, 4, 4),   3,          4,      "BasicBlock"),
    )
)

hrnet_w44 = tuple(
    StageSpec(num_channels=nc, num_blocks=nbl, num_modules=nm, num_branches=nbr, block=b)
    for (nc, nbl, nm, nbr, b) in (
        ((64),               (4),            1,          1,      "Bottleneck"),
        ((44, 88),           (4, 4),         1,          2,      "BasicBlock"),
        ((44, 88, 176),      (4, 4, 4),      4,          3,      "BasicBlock"),
        ((44, 88, 176, 352), (4, 4, 4, 4),   3,          4,      "BasicBlock"),
    )
)

hrnet_w48 = tuple(
    StageSpec(num_channels=nc, num_blocks=nbl, num_modules=nm, num_branches=nbr, block=b)
    for (nc, nbl, nm, nbr, b) in (
        ((64),               (4),            1,          1,      "Bottleneck"),
        ((48, 96),           (4, 4),         1,          2,      "BasicBlock"),
        ((48, 96, 192),      (4, 4, 4),      4,          3,      "BasicBlock"),
        ((48, 96, 192, 384), (4, 4, 4, 4),   3,          4,      "BasicBlock"),
    )
)

hrnet_w64 = tuple(
    StageSpec(num_channels=nc, num_blocks=nbl, num_modules=nm, num_branches=nbr, block=b)
    for (nc, nbl, nm, nbr, b) in (
        ((64),                (4),            1,          1,      "Bottleneck"),
        ((64, 128),           (4, 4),         1,          2,      "BasicBlock"),
        ((64, 128, 256),      (4, 4, 4),      4,          3,      "BasicBlock"),
        ((64, 128, 256, 512), (4, 4, 4, 4),   3,          4,      "BasicBlock"),
    )
)




class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, norm_layer, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self._norm_layer = norm_layer

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, norm_layer=self._norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], norm_layer=self._norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        self._norm_layer(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                       3, 2, 1, bias=False),
                                self._norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                       3, 2, 1, bias=False),
                                self._norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNet(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super(HRNet, self).__init__()


        self.fuze_method = "SUM"
        self.stage_specs = sys.modules[__name__].__getattribute__(model_cfg.arch)
        self._norm_layer = ops.NORM_LAYERS[model_cfg.norm_layer]
        self.multiscale  = model_cfg.multi_scale_output

        self.inplanes = 64

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = self._norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = self._norm_layer(64)
        self.relu = nn.ReLU(inplace=True)


        self.stage1_cfg = self.stage_specs[0]
        num_channels = self.stage1_cfg.num_channels
        block = getattr(res_blocks, self.stage1_cfg.block)
        num_blocks = self.stage1_cfg.num_blocks
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        # stage1_out_channel = block.expansion*num_channels
        # self.layer1 = self._make_layer(Bottleneck, self.inplanes, 64, 4)

        self.stage2_cfg = self.stage_specs[1]
        num_channels = self.stage2_cfg.num_channels
        block = getattr(res_blocks, self.stage2_cfg.block)
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = self.stage_specs[2]
        num_channels = self.stage3_cfg.num_channels
        block = getattr(res_blocks, self.stage3_cfg.block)
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = self.stage_specs[3]
        num_channels = self.stage4_cfg.num_channels
        block = getattr(res_blocks, self.stage4_cfg.block)
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=self.multiscale)
        self.num_channels = pre_stage_channels

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        Conv2d(num_channels_pre_layer[i],
                               num_channels_cur_layer[i],
                               3,
                               1,
                               1,
                               bias=False),
                        self._norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    # authors fuck TorchScript
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        self._norm_layer(outchannels),
                        nn.ReLU(inplace=True)))

                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                blocks.conv1x1(inplanes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self._norm_layer))
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(inplanes, planes, norm_layer=self._norm_layer))

        return nn.Sequential(*layers)


    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules  = layer_config.num_modules
        num_branches = layer_config.num_branches
        num_blocks   = layer_config.num_blocks
        num_channels = layer_config.num_channels
        block = getattr(res_blocks, layer_config.block)

        # All original configs have 'FUSE_METHOD' = 'SUM'
        fuse_method = self.fuze_method #layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     self._norm_layer,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    @property
    def feature_channels(self):
        if self.multiscale:
            return self.stage_specs[-1].num_channels
        else:
            return self.stage_specs[-1].num_channels[-1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg.num_branches):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.num_branches):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return tuple(y_list)
