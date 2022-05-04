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

def sgd_step(dt, bs, xtr, ytr, loss, model, gen, replacement=False):
    if replacement:
        index = torch.randint(len(xtr), (bs,), generator=gen)
    else:
        index = torch.randperm(len(xtr), generator=gen)[:bs]

    x = xtr[index]
    y = ytr[index]

    y_pred = model(x)
    loss_batch = loss(y_pred, y)
    # print(loss_batch)
    loss_batch.retain_grad()

    # for p in model.parameters():
    #     print(f"{p.grad}")

    model.zero_grad()
    # print(f"{loss_batch.item()}")

    loss_batch.backward()
    grad_norm = 0
    with torch.no_grad():
        for p in model.parameters():
            p.add_(-dt * p.grad)
            grad_norm += p.grad.data.norm() ** 2
            # print(p.grad)

    # grad = torch.autograd.grad(loss_batch,model.parameters())
    # grad_norm = 0
    # with torch.no_grad():
    #     for p,g in zip(model.parameters(), grad):
    #         p.add_(-dt * g)
    #         grad_norm += g.norm() ** 2
    
    return model, grad_norm


def train_model(xtr, ytr, xte, yte, loss_type, model, replacement, **args):
    gen = torch.Generator(device="cpu").manual_seed(args['seed_batch'])
    loss = loss_fun(loss_type)

    max_steps = 100000
    checkpoint_steps = 1000

    ckpt_step = 0
    for steps in range(max_steps):
        model, grad_norm = sgd_step(args['dt'],args['bs'], xtr, ytr, loss, model, gen, replacement)
        
        if steps - ckpt_step > checkpoint_steps:
            ckpt_step = steps
            yield steps, model, grad_norm, args['dt']*steps
            # yield the predictor and other quantities every checkpoint_steps steps
    


    
 
  
 
    
  
