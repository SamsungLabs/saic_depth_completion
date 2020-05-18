import torch

import torch.nn.functional as F

from saic_depth_completion.utils import registry
# refactor this to
class MetaModel(torch.nn.Module):
    def __init__(self, cfg, device):
        super(MetaModel, self).__init__()
        self.model = registry.MODELS[cfg.model.arch](cfg.model)
        self.model.to(device)
        self.device= device

        self.rgb_mean = cfg.train.rgb_mean
        self.rgb_std = cfg.train.rgb_std

        self.depth_mean = cfg.train.depth_mean
        self.depth_std = cfg.train.depth_std

    def forward(self, batch):
        return self.model(batch)

    def preprocess(self, batch):

        batch["color"] = batch["color"] - torch.tensor(self.rgb_mean).view(1, 3, 1, 1)
        batch["color"] = batch["color"] / torch.tensor(self.rgb_std).view(1, 3, 1, 1)

        mask = batch["raw_depth"] != 0
        batch["raw_depth"][mask] = batch["raw_depth"][mask] - self.depth_mean
        batch["raw_depth"][mask] = batch["raw_depth"][mask] / self.depth_std

        for k, v in batch.items():
            batch[k] = v.to(self.device)

        return batch

    def postprocess(self, input):
        return self.model.postprocess(input)
    def criterion(self, pred, gt):
        return self.model.criterion(pred, gt)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)