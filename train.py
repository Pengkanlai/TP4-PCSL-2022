import torch
import numpy as np

def mean_squared_loss(y, y_pred):
    msl = pow(y_pred-y,2).mean()
    return msl

def gradient_MSL_linear_model(X, y, y_pred):
    grad = -2*X*(y_pred-y)
    return grad  
  
def gradient_MSL_diagonal_linear_model():
  
def run_SGD_linear_model_regression(X, y, learning_rate, nb_iterations, batch_size):
    W = torch.random.normal(0., 1., size = (len(X[0])))
    y_pred = model_linear(X,W)
    train_loss = []
    
    for i in range(nb_iterations):
        grad = 0
        for mu in range(len(y)):
            grad += gradient_MSL_linear_model(X[mu], y[mu], y_pred[mu])
        W -= learning_rate * grad 
        
        y_pred = model_linear(X,W) 
        train_loss[i] = mean_squared_loss(y, y_pred)
        
    return W  
  
  def hinge_loss(y,y_pred):
    hinge_loss = (1 - torch.sign(max(0,y*y_pred))).mean()
    return hinge_loss
  
  def gradient_hinge_loss():
    
  def run_SGD_diagonal_linear_model_regression(X, y, learning_rate, nb_iterations, batch_size):
    W_1 = torch.random.normal(0., 1., size = (len(X[0])))
    W_2 = torch.random.normal(0., 1., size = (len(X[0])))
    y_pred = model_diagonal_linear(X, W_1, W_2)
    train_loss = []
    
    for i in range(nb_iterations):
        grad = 0
        for mu in range(len(y)):
            grad += gradient_MSL_diagonal_linear_model(X[mu], y[mu], y_pred[mu])
        W -= learning_rate * grad 
        
        y_pred = model_diagonal_linear(X, W_1, W_2)
        train_loss[i] = mean_squared_loss(y, y_pred)
        
    return W     
