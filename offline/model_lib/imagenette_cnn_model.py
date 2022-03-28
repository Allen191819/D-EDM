import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, gpu=True):
        super(Model, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(128 * 16 * 16, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)

        if gpu:
            self.to(device)

    def forward(self, x):
        if self.gpu:
            x = x.to(device)
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv3(x)))
        x = F.relu(self.linear(x.view(B, 128 * 16 * 16)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.to(device)
        return F.cross_entropy(pred, label)
