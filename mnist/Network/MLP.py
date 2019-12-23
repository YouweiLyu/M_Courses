import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

class MLP(nn.Module):
    def __init__(self, bn):
        super(MLP, self).__init__()
        if bn:
            self.fc = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(28*28, 200),
                nn.ReLU(),
                nn.Linear(200, 10),
                nn.Sigmoid()
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        return out