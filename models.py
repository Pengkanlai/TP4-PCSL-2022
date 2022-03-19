import torch
import numpy as np

def linear(X,W):
    return X@W.t()

def diagonal_linear(X, W_1, W_2):
    return (W_1*W_2)@X
