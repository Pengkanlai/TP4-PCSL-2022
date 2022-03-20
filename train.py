import torch
import numpy as np

def loss(type,y,y_pred):
    if type == 'mean_squared':
        loss = pow(y_pred-y,2).mean()
    elif type == 'hinge':
        loss = max(0, 1-torch.sign(y)*y_pred).mean()
    return loss

def sgd_step(dt,bs,xtr,ytr,loss_type,model):
    index = np.random.choice(xtr.size(0),bs,replace=False)
    x = xtr[index]
    y = ytr[index]
    if model == 'linear':
        y_pred = linear(x,w_1)
        Loss = loss(loss_type,y,y_pred)
        for i in range(w_1.size(0)):
            w_1 -= dt * torch.autograd.grad(Loss,w_1)
    if model == 'diagonal_linear':
        y_pred = diagonal_linear(x,w_1,w_2)
        Loss = loss(loss_type,y,y_pred)
        for i in range(w_1.size(0)):
            w_1 -= dt * torch.autograd.grad(Loss,(w_1,w_2))[0]
            w_2 -= dt * torch.autograd.grad(Loss,(w_1,w_2))[1]

def train_model(dt,bs,xtr,ytr,loss_type,model):
    nb_iterations = 1000
    data = []
    w_1 = torch.empty(xtr.size(0)).normal_(0,1)
    w_2 = torch.empty(xtr.size(0)).normal_(0,1)
    for i in range(nb_iterations):
        output = sgd_step(dt,bs,xtr,ytr,loss_type,model)
        if model == 'linear':
            y_pred = linear(x,w_1)
            Ltr = loss(loss_type,ytr,y_pred)
        if model == 'diagonal_linear':
            y_pred = diagonal_linear(x,w_1,w_2)
            Ltr = loss(loss_type,ytr,y_pred)
        data.append(Ltr)
    return data
    
    
    
 
  
 
    
  
