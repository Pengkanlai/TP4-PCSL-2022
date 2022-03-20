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
        Loss.backward()
        for i in range(w_1.size(0)):
            w_1[i] -= dt * w_1[i].grad
    elif model == 'diagonal_linear':
        y_pred = diagonal_linear(x,w_1,w_2)
        Loss = loss(loss_type,y,y_pred)
        Loss.backward()
        for i in range(w_1.size(0)):
            w_1[i] -= dt * w_1[i].grad
            w_2[i] -= dt * w_2[i].gra

def train_model(dt,bs,xtr,ytr,loss_type,model):
    nb_iterations = 1000
    data = []
    w_1 = torch.empty(xtr.size(1)).normal_(0,1).requires_grad_()
    w_2 = torch.empty(xtr.size(1)).normal_(0,1).requires_grad_()
    for i in range(nb_iterations):
        w_1.grad.data.zero_()
        w_2.grad.data.zero_()
        sgd_step(dt,bs,xtr,ytr,loss_type,model)
        if model == 'linear':
            y_pred = linear(x,w_1)
            Ltr = loss(loss_type,ytr,y_pred)
        elif model == 'diagonal_linear':
            y_pred = diagonal_linear(x,w_1,w_2)
            Ltr = loss(loss_type,ytr,y_pred)
        data.append(Ltr)
    return data
    
    
    
 
  
 
    
  
