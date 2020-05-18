import os
import re
import torch
from torch.hub import load_state_dict_from_url


model_resnet_imagenet = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

hrnet_root = "/dbstore/datasets/HRNet-Image-Classification_weights"
model_hrnet_imagenet = {
    "hrnet_w18": os.path.join(hrnet_root, "hrnetv2_w18_imagenet_pretrained.pth"),
    "hrnet_w18_small_v1": os.path.join(hrnet_root, "hrnet_w18_small_model_v1.pth"),
    "hrnet_w18_small_v2": os.path.join(hrnet_root, "hrnet_w18_small_model_v2.pth"),
    "hrnet_w30": os.path.join(hrnet_root, "hrnetv2_w30_imagenet_pretrained.pth"),
    "hrnet_w32": os.path.join(hrnet_root, "hrnetv2_w32_imagenet_pretrained.pth"),
    "hrnet_w40": os.path.join(hrnet_root, "hrnetv2_w40_imagenet_pretrained.pth"),
    "hrnet_w44": os.path.join(hrnet_root, "hrnetv2_w44_imagenet_pretrained.pth"),
    "hrnet_w48": os.path.join(hrnet_root, "hrnetv2_w48_imagenet_pretrained.pth"),
    "hrnet_w64": os.path.join(hrnet_root, "hrnetv2_w64_imagenet_pretrained.pth"),
}


model_efficientnet_imagenet = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}

def _load_state_dict_hrnet(key):
    state_dict = torch.load(model_hrnet_imagenet[key])
    return state_dict

def _load_state_dict_resnet(key):
    state_dict = load_state_dict_from_url(model_resnet_imagenet[key], progress=True)
    return state_dict

def _load_state_dict_efficientnet(key):
    state_dict = load_state_dict_from_url(model_efficientnet_imagenet[key], progress=True)
    return state_dict