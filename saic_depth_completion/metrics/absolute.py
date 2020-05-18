import torch
from torch import nn

###### LOSSES #######

class BerHuLoss(nn.Module):
    def __init__(self, scale=0.5, eps=1e-5):
        super(BerHuLoss, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        img1 = img1[img2 > self.eps]
        img2 = img2[img2 > self.eps]

        diff = torch.abs(img1 - img2)
        threshold = self.scale * torch.max(diff).detach()
        mask = diff > threshold
        diff[mask] = ((img1[mask]-img2[mask])**2 + threshold**2) / (2*threshold + self.eps)
        return diff.sum() / diff.numel()


class LogDepthL1Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super(LogDepthL1Loss, self).__init__()
        self.eps = eps
    def forward(self, pred, gt):
        mask = gt > self.eps
        diff = torch.abs(torch.log(gt[mask]) - pred[mask])
        return diff.mean()

###### METRICS #######

class DepthL1Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super(DepthL1Loss, self).__init__()
        self.eps = eps
    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        mask = gt > self.eps
        img1[~mask] = 0.
        img2[~mask] = 0.
        return nn.L1Loss(reduction="sum")(img1, img2), pred.numel()

class DepthL2Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super(DepthL2Loss, self).__init__()
        self.eps = eps
    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        mask = gt > self.eps
        img1[~mask] = 0.
        img2[~mask] = 0.
        return nn.MSELoss(reduction="sum")(img1, img2), pred.numel()
