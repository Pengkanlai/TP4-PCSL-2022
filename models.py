import torch
import numpy as np

def train_model(model,X,W_1,W_2):
    if model == 'linear':
        f = X @ W_1.t()
    elif model == 'diagonal_linear':
        f = (W_1*W_2)@X
    return f
