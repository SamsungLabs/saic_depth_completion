from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

###### METRICS #######

class DepthRel(nn.Module):
    def __init__(self, eps=1e-5):
        super(DepthRel, self).__init__()
        self.eps = eps
    def forward(self, pred, gt):
        mask = gt > self.eps
        diff = torch.abs(gt[mask] - pred[mask]) / gt[mask]
        return diff.median()

class Miss(nn.Module):
    def __init__(self, thresh, eps=1e-5):
        super(Miss, self).__init__()
        self.thresh = thresh
        self.eps = eps
    def forward(self, pred, gt):
        mask = (gt > self.eps)# & (pred > self.eps)

        pred_over_gt, gt_over_pred = pred[mask] / gt[mask], gt[mask] / pred[mask]
        miss_map = torch.max(pred_over_gt, gt_over_pred)
        hit_rate = torch.sum(miss_map < self.thresh ).float()#, miss_map.numel()

        # if torch.isnan(hit_rate):return 0

        return hit_rate, miss_map.numel()


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, eps=1e-5):
        super(SSIM, self).__init__()
        self.eps = eps
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred, gt):

        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        img2[img2 < self.eps] = 0
        img1[img2 < self.eps] = 0

        (_, channel, _, _) = img1.size()


        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
