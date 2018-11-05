#!/usr/bin/env python3

import numpy as np

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================


def sigmoid(x):
    """Sigmoidal activation function.

    Args:
        x (ndarray): weighted sum of inputs
    """
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)


def identity(x):
    """Identity activation function. Input equals output.

    Args:
        x (ndarray): weighted sum of inputs.
    """
    return x


def identity_derivative(x):
    """Simply returns the derivative, 1, of the identity."""
    return 1.0


def softmax(x):
    """The Softmax activation function. Assures that no outliers can 
    dominate too much.

    Args:
        x (ndarray): weighted sum of inputs.
    """
    z_exp = np.exp(x)
    z_exp_sum = np.sum(z_exp)
    return z_exp/z_exp_sum


def softmax_derivative(x):
    """The derivative of the Softmax activation function.

    Args:
        x (ndarray): weighted sum of inputs.
    """
    raise ValueError("softmax_derivative not intended for hidden layers.")
    z_exp = softmax(x)
    z_id = np.einsum("i,j->ij", z_exp, np.eye(3))
    z_ij = np.einsum("i,j->ij", z_exp, z_exp)
    return z_id - z_ij

def heaviside(x):
    """The Heaviside activation function.

    Args:
        x (ndarray): weighted sum of inputs
    """
    return np.where(x >= 0, 1, 0)


def heaviside_derivative(x):
    """The derivative of the Heaviside activation function.

    Args:
        x (ndarray): weighted sum of inputs
    """
    return np.zeros(x.shape)


def relu(x):
    """The rectifier activation function. Only activates if argument x is 
    positive.

    Args:
        x (ndarray): weighted sum of inputs
    """
    return np.where(x >= 0, x, 0)


def relu_derivative(x):
    """The derivative of the tangens hyperbolicus activation function.

    Args:
        x (ndarray): weighted sum of inputs
    """
    return np.where(x >= 0, 1, 0)


def tanh(x):
    """The tangens hyperbolicus activation function.

    Args:
        x (ndarray): weighted sum of inputs
    """
    return np.tanh(x)


def tanh_derivative(x):
    """The derivative of the tangens hyperbolicus activation function.

    Args:
        x (ndarray): weighted sum of inputs
    """
    return 1/(np.cosh(x)**2)


# =============================================================================
# COST FUNCTIONS
# =============================================================================


def mse_cost(x, y):
    """Mean Square error cost function."""
    return np.sum((x - y)**2)/2  # *y.shape[0])


def mse_cost_derivative(x, y):
    """Mean Square error cost function derivative."""
    return (x - y)

def log_entropy(x, y):
    pass

def log_entropy_derivative(x, y):
    pass