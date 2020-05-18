import sys
from collections import namedtuple

from torch import nn

from saic_depth_completion.modeling.backbone import res_blocks
from saic_depth_completion import ops

StageSpec = namedtuple("StageSpec", ["block_count", "block"],)

resnet18 = tuple(StageSpec(block_count=c, block=b)
    for (c, b) in ((2, "BasicBlock"),(2, "BasicBlock"),(2, "BasicBlock"),(2, "BasicBlock"))
)

resnet34 = tuple(StageSpec(block_count=c, block=b)
    for (c, b) in ((3, "BasicBlock"),(4, "BasicBlock"),(6, "BasicBlock"),(3, "BasicBlock"))
)

resnet50 = tuple(StageSpec(block_count=c, block=b)
    for (c, b) in ((3, "Bottleneck"),(4, "Bottleneck"),(6, "Bottleneck"),(3, "Bottleneck"))
)

resnet101 = tuple(StageSpec(block_count=c, block=b)
    for (c, b) in ((3, "Bottleneck"),(4, "Bottleneck"),(23, "Bottleneck"),(3, "Bottleneck"))
)

resnet152 = tuple(StageSpec(block_count=c, block=b)
    for (c, b) in ((3, "Bottleneck"),(8, "Bottleneck"),(36, "Bottleneck"),(3, "Bottleneck"))
)

class ResNet(nn.Module):

    def __init__(self, model_cfg, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):

        super(ResNet, self).__init__()

        self.stage_specs    = sys.modules[__name__].__getattribute__(model_cfg.arch)
        self.block          = getattr(res_blocks, self.stage_specs[0].block)
        self._norm_layer    = ops.NORM_LAYERS[model_cfg.norm_layer]
        self.multiscale     = model_cfg.multi_scale_output
        self.base_channel   = 64 * self.block.expansion
        self.input_channels = 3


        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 1/4
        self.layer1 = self._make_layer(self.block, 64, self.stage_specs[0].block_count)
        # 1/8
        self.layer2 = self._make_layer(self.block, 128, self.stage_specs[1].block_count, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        # 1/16
        self.layer3 = self._make_layer(self.block, 256, self.stage_specs[2].block_count, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # 1/32
        self.layer4 = self._make_layer(self.block, 512, self.stage_specs[3].block_count, stride=2,
                                       dilate=replace_stride_with_dilation[2])


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        # self.cuda()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                res_blocks.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    @property
    def feature_channels(self):
        if self.multiscale:
            return self.base_channel, self.base_channel*2, \
                   self.base_channel*4, self.base_channel*8
        return self.base_channel*8

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        res = []
        x = self.layer1(x)
        res += [x]
        x = self.layer2(x)
        res += [x]
        x = self.layer3(x)
        res += [x]
        x = self.layer4(x)
        res += [x]

        if self.multiscale:
            return tuple(res)
        return tuple([x])
