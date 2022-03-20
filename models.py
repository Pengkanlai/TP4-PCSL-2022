import torch
import numpy as np

def linear(x,w):
    return x @ w

def diagonal_linear(x,w_1,w_2):
    return (w_1*w_2)@x
