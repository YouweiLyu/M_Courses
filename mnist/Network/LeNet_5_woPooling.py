import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

class LeNet_5_woPooling(nn.Module):
    def __init__(self, bn):
        super(LeNet_5_woPooling, self).__init__()
        if bn:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 20, 5, stride=1, padding=0),
                nn.BatchNorm2d(20),
                nn.ReLU(),
                nn.Conv2d(20, 16, 5, stride=2, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                )
            self.fc = nn.Sequential(
                nn.Linear(1600, 120),
                nn.BatchNorm1d(120),
                nn.ReLU(inplace=True),
                nn.Linear(120, 84),
                nn.BatchNorm1d(84),
                nn.ReLU(inplace=True),
                nn.Linear(84, 10),
                nn.Sigmoid()
                )
        else: 
            self.conv = nn.Sequential(
                nn.Conv2d(1, 20, 5, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(20, 16, 5, stride=2, padding=0),
                nn.ReLU(),
                )
            self.fc = nn.Sequential(
                nn.Linear(1600, 120),
                nn.ReLU(inplace=True),
                nn.Linear(120, 84),
                nn.ReLU(inplace=True),
                nn.Linear(84, 10),
                nn.Sigmoid()
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)


    def forward(self, x):
        out_conv = self.conv(x)
        out = self.fc(out_conv.view(out_conv.size(0), -1))
        return out
