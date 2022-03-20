import numpy as np
import torch

d = 1000
n = 20

covariance_V = torch.cat((torch.ones(10),torch.ones(d-10)*0.1))

xtr = torch.randn((n,d))*covariance_V
xte = torch.randn((n,d))*covariance_V

beta = torch.zeros(d)
beta[0] = 1
  
ytr = beta@xtr.T
yte = beta@xte.T
