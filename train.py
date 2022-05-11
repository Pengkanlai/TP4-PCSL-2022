import torch
from torch.nn.functional import relu
import numpy as np
from functools import partial

def mse(alpha, y_pred, y):
    return pow(alpha * y_pred-y,2).mean() / alpha

def hinge(alpha, y_pred, y):
    return relu(1- alpha * y_pred * y).mean() / alpha

def loss_fun(alpha, type):
    if type == 'mean_squared':
        return partial(mse, alpha)
    elif type == 'hinge':
        return partial(hinge, alpha)

def sgd_step(dt, bs, xtr, ytr, otr0, loss, model, gen, replacement=False):
    if replacement:
        index = torch.randint(len(xtr), (bs,), generator=gen)
    else:
        index = torch.randperm(len(xtr), generator=gen)[:bs]

    x = xtr[index]
    y = ytr[index]
    o0 = otr0[index]

    y_pred = model(x) - o0
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


def train_model(xtr, ytr, otr0, xte, yte, ote0, loss, model, replacement, args):
    gen = torch.Generator(device="cpu").manual_seed(args['seed_batch'])

    max_steps = 100000
    
    checkpoint_steps = 1
    ckpt_step = 0
    for steps in range(1,max_steps+1):
        model, grad_norm = sgd_step(args['dt'],args['bs'], xtr, ytr, otr0, loss, model, gen, replacement)
        
        if steps - ckpt_step >= checkpoint_steps:
            ckpt_step = steps
            checkpoint_steps = 1 if steps <100 else int(steps/2)
            yield steps, args['dt']*steps, model, grad_norm.item()
            # yield the predictor and other quantities every checkpoint_steps steps