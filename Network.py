import torch
import torch.nn.functional as F
from constants import *


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 32, 3)
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.conv4 = torch.nn.Conv2d(64, 64, 3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.linear1 = torch.nn.Linear(1600, 120)
        self.linear2 = torch.nn.Linear(120, 64)
        self.linear3 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x
