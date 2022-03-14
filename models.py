import torch
import numpy as np

def model_linear(X,W):
    return X @ W.t()

def model_diagonal_linear(X, W_1, W_2):
    return torch.mul(W_1,W_2.t()).mean()*X
