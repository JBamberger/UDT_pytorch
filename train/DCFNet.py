import torch
import torch.fft as fft
import torch.nn as nn

from train.DCFNetFeature import DCFNetFeature


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.yf = config.yf.clone()
        # regularization parameter in the regression formulation
        self.lambda0 = config.lambda0

    def forward(self, template, search, label):
        # Template shape: R[batch, 32, 121, 121]
        # Search shape:   R[batch, 32, 121, 121]
        # Label shape:    R[batch,  1, 121,  61]

        # zf & xf shape: C[batch, 32, 121, 61]
        zf = fft.rfftn(template, dim=[-2, -1])
        xf = fft.rfftn(search, dim=[-2, -1])

        # R[batch, 1, 121, 61]
        kzzf = torch.sum(zf.real ** 2 + zf.imag ** 2, dim=1, keepdim=True)

        # C[batch, 1, 121, 61]
        t = xf * torch.conj(zf)
        kxzf = torch.sum(t, dim=1, keepdim=True)

        # C[batch, 1, 121, 61]
        alphaf = label.to(device=template.device) / (kzzf + self.lambda0)

        # R[batch, 1, 121, 121]
        response = fft.irfftn(kxzf * alphaf, s=[121, 121], dim=[-2, -1])
        return response


if __name__ == '__main__':
    # network test
    net = DCFNet()
    net.eval()
