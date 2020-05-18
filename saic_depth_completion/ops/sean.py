import torch
from torch import nn
from torch.nn import functional as F

class SEAN(nn.Module):

    def __init__(self, x_ch, y_ch, kernel_size=3, upsample='nearest'):
        super(SEAN, self).__init__()
        assert upsample in ['nearest', 'bilinear']
        self.upsample = upsample

        padding = (kernel_size - 1) // 2

        self.gamma_y = nn.Conv2d(
            y_ch, x_ch, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.beta_y = nn.Conv2d(
            y_ch, x_ch, kernel_size=kernel_size, padding=padding, bias=False
        )

        self.gamma_x = nn.Conv2d(
            x_ch, x_ch, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.beta_x = nn.Conv2d(
            x_ch, x_ch, kernel_size=kernel_size, padding=padding, bias=False
        )

        self.w_gamma = torch.tensor(1., requires_grad=True).cuda()
        self.w_beta = torch.tensor(1., requires_grad=True).cuda()


        # we assume that there is a some distribution at each cell in tensor
        # => we need to compute stats over batch only
        self.bn = nn.BatchNorm2d(x_ch, affine=False)

        # self.cuda()

    def forward(self, x, y):

        y = F.interpolate(y, size=x.size()[-2:], mode=self.upsample)

        x_normalized = self.bn(x)

        # do not need relu !!! We should be able to sub from signal
        gamma_y = self.gamma_y(y)
        beta_y = self.beta_y(y)

        gamma_x = self.gamma_x(x)
        beta_x = self.beta_x(x)


        gamma = (1 - self.w_gamma) * gamma_x + self.w_gamma * gamma_y
        beta = (1 - self.w_beta) * beta_x +  self.w_beta * beta_y

        return  (gamma) * x_normalized + beta