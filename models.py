import torch
import numpy as np

def model_linear(X,W):
    return X @ W.t()
