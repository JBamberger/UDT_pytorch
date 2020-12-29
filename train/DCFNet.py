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
        # z shape: batch, 32, 121, 121
        # x shape: batch, 32, 121, 121
        # Label shape: batch, 1, 121, 61, 2


        # z = self.feature(z)
        # x = self.feature(x)

        # shapes: batch, 32, 121, 61, 2
        zf = torch.rfft(z, signal_ndim=2)
        xf = torch.rfft(x, signal_ndim=2)

        # shape change: [batch, 32, 121, 61, 2] -> [batch, 32, 121, 61, 1] -> [batch, 1, 121, 61, 1]
        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)

        # shape change: [batch, 32, 121, 61, 2] -> [batch, 32, 121, 61, 2] -> [batch, 1, 121, 61, 2]
        kxzf = torch.sum(cn.mulconj(xf, zf), dim=1, keepdim=True)

        # shape: [batch, 1, 121, 61, 2]
        alphaf = label.to(device=z.device) / (kzzf + self.lambda0)
        # alphaf = self.yf.to(device=z.device) / (kzzf + self.lambda0) # very Ugly

        # shape change: -> [42, 1, 121, 61, 2] -> [42, 1, 121, 121]
        return torch.irfft(cn.mul(kxzf, alphaf), signal_ndim=2)


if __name__ == '__main__':
    # network test
    net = DCFNet()
    net.eval()
