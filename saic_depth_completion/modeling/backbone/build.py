from saic_depth_completion.modeling.backbone.resnet import ResNet
from saic_depth_completion.modeling.backbone.hrnet import HRNet
from saic_depth_completion.modeling.backbone.efficientnet import EfficientNet
from saic_depth_completion.utils import registry
from saic_depth_completion.utils.model_zoo import (_load_state_dict_hrnet,
                                                   _load_state_dict_resnet,
                                                   _load_state_dict_efficientnet)


@registry.BACKBONES.register("resnet18")
@registry.BACKBONES.register("resnet34")
@registry.BACKBONES.register("resnet50")
@registry.BACKBONES.register("resnet101")
@registry.BACKBONES.register("resnet152")
def build_resnet(cfg):
    resnet = ResNet(cfg)
    if cfg.imagenet is True:
        state_dict = _load_state_dict_resnet(cfg.arch)
        resnet.load_state_dict(state_dict, strict=False)
    return resnet

@registry.BACKBONES.register("hrnet_w18")
@registry.BACKBONES.register("hrnet_w18_small_v1")
@registry.BACKBONES.register("hrnet_w18_small_v2")
@registry.BACKBONES.register("hrnet_w30")
@registry.BACKBONES.register("hrnet_w32")
@registry.BACKBONES.register("hrnet_w40")
@registry.BACKBONES.register("hrnet_w44")
@registry.BACKBONES.register("hrnet_w48")
@registry.BACKBONES.register("hrnet_w64")
def build_hrnet(cfg):
    hrnet = HRNet(cfg)
    if cfg.imagenet is True:
        state_dict = _load_state_dict_hrnet(cfg.arch)
        hrnet.load_state_dict(state_dict, strict=False)
    return hrnet


@registry.BACKBONES.register("efficientnet-b0")
@registry.BACKBONES.register("efficientnet-b1")
@registry.BACKBONES.register("efficientnet-b2")
@registry.BACKBONES.register("efficientnet-b3")
@registry.BACKBONES.register("efficientnet-b4")
@registry.BACKBONES.register("efficientnet-b5")
@registry.BACKBONES.register("efficientnet-b6")
@registry.BACKBONES.register("efficientnet-b7")
def build_efficientnet(cfg):
    efficientnet = EfficientNet(cfg)
    if cfg.imagenet is True:
        state_dict = _load_state_dict_efficientnet(cfg.arch)
        efficientnet.load_state_dict(state_dict, strict=False)
    return efficientnet


def build_backbone(cfg):
    return registry.BACKBONES[cfg.arch](cfg)

