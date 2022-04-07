import torch
from torch.nn.functional import relu
import numpy as np

def mse(y_pred, y):
    return pow(y_pred-y,2).mean()

def hinge(y_pred, y):
    return relu(1-y_pred*y).mean()

def loss_fun(type):
    if type == 'mean_squared':
        return mse
    elif type == 'hinge':
        return hinge

def sgd_step(dt,bs,xtr,ytr,loss,model, gen, replacement=False):
    if replacement:
        index = torch.randint(len(xtr), (bs,), generator=gen)
    else:
        index = torch.randperm(len(xtr), generator=gen)[:bs]

    x = xtr[index]
    y = ytr[index]

    y_pred = model(x)
    loss_batch = loss(y_pred, y)
    
    model.zero_grad()
    loss_batch.backward()

    for p in model.parameters():
        p = p - dt * p.grad


def train_model(xtr,ytr,loss_type, model, replacement, **args):
    gen = torch.Generator(device="cpu").manual_seed(args['seed_batch'])
    loss = loss_fun(loss_type)
    nb_iterations = 1000
    data = []

    for i in range(nb_iterations):
        sgd_step(args['dt'],args['bs'], xtr, ytr, loss, model, gen, replacement)
        
        y_pred = model(xtr)
        Ltr = loss(y_pred, ytr)
        data.append(Ltr.item())
        # dictionary with all interesting observables
    return data
    
    # steps = 0
    # for i in range(checkpoint_steps):
    #     sgd_step(args['dt'],args['bs'], xtr, ytr, loss, model, gen, replacement)
        
    # yield model, steps+checkpoint_steps


    
 
  
 
    
  
