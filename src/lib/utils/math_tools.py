#!/usr/bin/env python3

import numpy as np

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================


def logistic(x):
    """Sigmoidal activation function.

    Args:
        x (ndarray): weighted sum of inputs
    """
    return 1.0/(1.0 + np.exp(-x))


def logistic_derivative(x):
    s = logistic(x)
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
    # np.clip(x, 0, np.finfo(x.dtype).max, out=x)
    # return x
    return np.where(x >= 0, x, 0)


def relu_derivative(x):
    """The derivative of the tangens hyperbolicus activation function.

    Args:
        x (ndarray): weighted sum of inputs
    """
    return np.where(relu(x) > 0, 1, 0)


def tanh_(x):
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
    return 1 - tanh_(x)**2


# =============================================================================
# COST FUNCTIONS
# =============================================================================


def mse_cost(y, y_true):
    """Mean Square error cost function."""
    return 0.5*np.sum((y - y_true)**2)  # /y.shape[0]


def mse_cost_derivative(y, y_true):
    """Mean Square error cost function derivative."""
    return (y - y_true)


def log_entropy(y, y_true):
    """Cross entropy cost function."""

    cost1 = y_true * np.log(y)
    cost2 = (1 - y_true) * np.log(1 - y)
    return - np.sum(cost1 + cost2)


def log_entropy_derivative(y, y_true):
    """Derivative of cross entropy cost function."""
    y = np.clip(y, 1e-10, 1 - 1e-10)
    return (y - y_true) / (y*(1 - y))


def exponential_cost(y, y_true, tau=0.1):
    """Exponential cost function."""
    return tau*np.exp(1/tau * np.sum((y-y_true)**2))


def exponential_cost_derivative(y, y_true, tau=0.1):
    """Exponential cost function gradient."""
    return 2/tau * (y-y_true)*exponential_cost(y, y_true, tau)
