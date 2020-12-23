import torch
import torch.nn as nn

import complex_numbers as cn
from train.DCFNetFeature import DCFNetFeature


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.yf = config.yf.clone()
        self.lambda0 = config.lambda0

    def forward(self, z, x, label):
        # z = self.feature(z)
        # x = self.feature(x)
        zf = torch.rfft(z, signal_ndim=2)
        xf = torch.rfft(x, signal_ndim=2)

        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kxzf = torch.sum(cn.mulconj(xf, zf), dim=1, keepdim=True)

        alphaf = label.to(device=z.device) / (kzzf + self.lambda0)
        # alphaf = self.yf.to(device=z.device) / (kzzf + self.lambda0) # very Ugly
        return torch.irfft(cn.mul(kxzf, alphaf), signal_ndim=2)


if __name__ == '__main__':
    # network test
    net = DCFNet()
    net.eval()
