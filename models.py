# from audioop import bias
import torch
from torch import nn


class testmod(nn.Module):
    def __init__(self, d):
        super(testmod, self).__init__()
        self.layer = nn.Linear(d,1)

    def forward(self, x):
        return self.layer(x)


class linear_model(nn.Module):
    def __init__(self, d, bias=False):
        super().__init__()
        if bias:
            # self.linear_stack = nn.Sequential(
            #     nn.Linear(d, 1, bias=True) # an affine operation: y = Wx + b
            # )
            self.linear_layer = nn.Linear(d, 1, bias=True) # an affine operation: y = Wx + b
            nn.init.normal_(self.linear_layer.weight, mean=0.0, std=1.0)
            nn.init.normal_(self.linear_layer.bias, mean=0.0, std=1.0)
        else:
            # self.linear_stack = nn.Sequential(
            #     nn.Linear(d, 1, bias=False) # an affine operation: y = Wx + b
            # )
            self.linear_layer = nn.Linear(d, 1, bias=False) # an homogenous linear operation: y = Wx
            nn.init.normal_(self.linear_layer.weight, mean=0.0, std=1.0)

    def forward(self, x):
        return self.linear_layer(x/(x.shape[1]**0.5)).flatten()


class diagonal_linear(nn.Module):
    def __init__(self, d, L, bias=False):
        super(diagonal_linear, self).__init__()
        self.L = L
        self.bias = bias
        Wplus = torch.randn(d)
        Wminus = torch.randn(d)
        self.Wplus = nn.Parameter(Wplus)
        self.Wminus = nn.Parameter(Wminus)
        
        if bias:
            self.b = nn.Parameter(torch.randn(1))
        else:
            self.b = None
        

    def forward(self, x):
        if self.bias:
            return (x/(x.shape[1]**0.5)) @ (self.Wplus**self.L - self.Wminus**self.L) + self.b
        else:
            return (x/(x.shape[1]**0.5)) @ (self.Wplus**self.L - self.Wminus**self.L)
