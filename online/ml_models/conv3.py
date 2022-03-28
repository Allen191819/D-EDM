"""conv3 module"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flatten the input"""

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv3(nn.Module):
    """conv3"""

    def __init__(
        self, x_dim: int = 28, n_channels: int = 3, n_classes: int = 10, mult: int = 2,
    ):
        self.downsample = False
        if x_dim > 32:
            x_dim = 32
            self.downsample = True
        linear_dim = int(x_dim / 4 - 3)
        linear_neurons = (linear_dim ** 2) * 64
        super(Conv3, self).__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 16 * mult, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16 * mult, 32 * mult, 3),
            nn.ReLU(),
            nn.Conv2d(32 * mult, 64 * mult, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(linear_neurons * mult, 256 * mult),
            nn.ReLU(),
            nn.Linear(256 * mult, self.n_classes),
        )

    def forward(self, x):
        if self.downsample:
            x = F.interpolate(x, 32)
        y = self.net(x)
        return y

