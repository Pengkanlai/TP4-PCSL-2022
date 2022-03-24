from audioop import bias
import torch
import torch.nn as nn


class linear(nn.Module):
    def __init__(self, d, bias=False):
        super(linear, self).__init__()
        if bias:
            self.linear_layer = nn.Linear(d, 1, bias=True) # an affine operation: y = Wx + b
        else:
            self.linear_layer = nn.Linear(d, 1, bias=False) # an homogenous linear operation: y = Wx

    def forward(self, x):
        x = self.linear_layer(x)
        return x


class diagonal_linear(nn.Module):
    def __init__(self, d):
        super(linear, self).__init__()
        self.linear1 = nn.Linear(d, 1, bias=False)
        self.linear2 = nn.Linear(d, 1, bias=False)
        self.linear3 = nn.Linear(d, 1, bias=False)
        for p1, p2, p3 in zip(self.linear1.parameters(), self.linear2.parameters(), self.linear3.paramteres()):
            p3 = p1*p2

    def forward(self, x):
        for p1, p2, p3 in zip(self.linear1.parameters(), self.linear2.parameters(), self.linear3.paramteres()):
            p3 = p1*p2
        x = self.linear3(x)
        return x
