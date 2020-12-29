from torch import nn as nn


class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        # input shape: batch, 3, h, w
        # output shape: batch, 32, h-4, w-4
        return self.feature(x)


if __name__ == '__main__':
    import torch

    feat = DCFNetFeature()
    feat.eval()

    batch = torch.zeros((32, 3, 125, 125))
    output = feat(batch)

    print(output.shape)
