#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt

#create noisy data
N = 100
slope = 3
intercept = 2
stdDev = 100 #standard deviation of noise
x = torch.linspace(-100,100,N) 
y = slope*x + intercept + np.random.normal(0, stdDev, N) #can use numpy for random

#Global fit parameters w and b, and the prediction vector yhat
w = torch.tensor(float(0),requires_grad = True) #slope
b = torch.tensor(float(-100),requires_grad = True) #intercept
yhat = torch.zeros(N) #prediction

#Learning parameters
wlr = 1e-4 #learning Rate of slope
blr = 2e-1 #learning Rate of bias
iterations = 4 # number of iterations

#general function which use and modify the global parameters
def forward(x):
    """Forward step is the prediction step. Where the input data (x) is multiplied
    by the parameters (w) and added to the bias (b), resulting in th prediction, yhat."""
    
    return x*w + b

def criterion(yhat, y):
    """Criterion is a measure of the error betwen the prediction (yhat) and the data (y).
    L2 error is likely the most common. Greater the power, the more weight to outliers. """
    
    return torch.mean( ( yhat - y )**2 )

def backward(loss, wlr, blr):
    """The backward step is the the optimization step. We calculate the gradient of the
    loss w.r.t. the model parameters (w and b) and travel in the negative gradient direction
    to minimize the loss. Simple Gradient Descent. """
    
    # tells the tree to calculate the parial derivates of the loss wrt all of the
    #contriubuting tensors with the "requires_grad = True" in their constructor.
    loss.backward() 
    
    #gradient descent (with different learning rates)
    w.data = w.data - wlr*w.grad.data
    b.data = b.data - blr*b.grad.data
    
    #must zero out the gradient otherwise pytorch accumulates the gradient. 
    w.grad.data.zero_()
    b.grad.data.zero_()

##lists to save the parameters and errors
params = []
error = []

for i in range(iterations):
    params.append([w.data,b.data,i]) # saving data
    
    yhat = forward(x) #major step 1/3
    
    loss = criterion(yhat, y) #major step 2/3

    error.append(loss.data) #saving data

    backward(loss, wlr, blr) #major step 3/3    

#saving data
params.append([w.data,b.data,iterations])
error.append(criterion(yhat, y).data)
params = np.array(params)
error = np.array(error)

#Simple display of the learning 
print(error)

plt.plot(x.numpy(), y.numpy(), 'k', label="data")
for param in params:
    plt.plot(x.numpy(),param[0]*x.numpy()+param[1], label = f'iter # {int(param[2])}')
plt.legend()
plt.title("simple linear regression with PyTorch")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('fig_LR_noDatasetClass.png')
