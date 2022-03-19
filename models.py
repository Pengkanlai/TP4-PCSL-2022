import torch
import numpy as np

def linear(X):
    W = torch.random.normal(0., 1., size = (len(X[0])))
    return X @ W.t()

def diagonal_linear(X):
    W_1 = torch.random.normal(0., 1., size = (len(X[0])))
    W_2 = torch.random.normal(0., 1., size = (len(X[0])))
    return (W_1*W_2)@X
