import torch
import numpy as np

nb_iterations = 100

def train_model(model,X):
    if model == 'linear':
        f = linear(X)
    elif model == 'diagonal_linear':
        f = diagonal_linear(X)
    

def loss(type,y,y_pred):
    if type == 'mean_squared':
        loss = pow(y_pred-y,2).mean()
    elif type == 'hinge':
        loss = max(0, 1-torch.sign(y)*y_pred).mean()
  
def SGD_step(f, xtr, ytr, dt, bs, loss_type):
    
    
    
    
 
  
 
    
  
