import torch
from torch import nn
from torch.nn import functional as F

class SPADE(nn.Module):

    def __init__(self, x_ch, y_ch, kernel_size=3, upsample='nearest'):
        super(SPADE, self).__init__()
        self.eps = 1e-5
        assert upsample in ['nearest', 'bilinear']
        self.upsample = upsample

        padding = (kernel_size) // 2

        self.gamma = nn.Conv2d(
            y_ch, x_ch, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.beta = nn.Conv2d(
            y_ch, x_ch, kernel_size=kernel_size, padding=padding, bias=False
        )

        # we assume that there is a some distribution at each cell in tensor
        # => we need to compute stats over batch only
        self.bn = nn.BatchNorm2d(x_ch, affine=False)

    def forward(self, x, y):

        y = F.interpolate(y, size=x.size()[-2:], mode=self.upsample)

        x_normalized = self.bn(x)

        # do not need relu !!! We should be able to sub from signal
        gamma = self.gamma(y)
        beta = self.beta(y)

        return  (1+gamma) * x_normalized + beta


class SelfSPADE(nn.Module):

    def __init__(self, x_ch, y_ch, kernel_size=3, upsample='nearest'):
        super(SelfSPADE, self).__init__()
        self.eps = 1e-5
        assert upsample in ['nearest', 'bilinear']
        self.upsample = upsample

        padding = (kernel_size) // 2

        self.gamma = nn.Conv2d(
            y_ch+x_ch, x_ch, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.beta = nn.Conv2d(
            y_ch+x_ch, x_ch, kernel_size=kernel_size, padding=padding, bias=False
        )

        self.adapt = nn.Conv2d(
            y_ch+x_ch, x_ch, kernel_size=1, padding=0, bias=False
        )

        # we assume that there is a some distribution at each cell in tensor
        # => we need to compute stats over batch only
        self.bn = nn.BatchNorm2d(x_ch, affine=False)


    def forward(self, x, y):

        y = F.interpolate(y, size=x.size()[-2:], mode=self.upsample)

        x = torch.cat([x, y], dim=1)

        x_normalized = self.bn(self.adapt(x))

        # do not need relu !!! We should be able to sub from signal
        gamma = 0.1 * self.gamma(x)
        beta = 0.1 * self.beta(x)

        return  (gamma) * x_normalized + beta
