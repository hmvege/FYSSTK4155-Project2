#!/usr/bin/env python3

import numpy as np


#### ACTIVATION FUNCTIONS
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)

def identity(x):
    return x

def identity_derivative(x):
    return 1.0

def softmax(x):
    pass

def softmax_derivative(x):
    pass

def heaviside(x):
    pass

def heaviside_derivative(x):
    pass

def tanh(x):
    pass

def tanh_derivative(x):
    pass

#### COST FUNCTIONS
def mse_cost(x, y):
    """Mean Square error cost function."""
    return np.sum((x - y)**2)/2  # *y.shape[0])

def mse_cost_derivative(x, y):
    """Mean Square error cost function derivative."""
    return (x - y)
