#!/usr/bin/env python3

import numpy as np
from scipy.special import expit

AVAILABLE_ACTIVATIONS = ["identity", "sigmoid", "relu", "tanh", "heaviside"]

AVAILABLE_OUTPUT_ACTIVATIONS = [
    "identity", "sigmoid", "softmax"]

AVAILABLE_COST_FUNCTIONS = ["mse", "log_loss", "exponential_cost",
                            "hellinger_distance",
                            "kullback_leibler_divergence",
                            "generalized_kullback_leibler_divergence",
                            "itakura_saito_distance"]

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================


def sigmoid(x):
    """Sigmoidal activation function.

    Args:
        x (ndarray): weighted sum of inputs
    """
    return expit(x)
    # return 1.0/(1.0 + np.exp(-x))


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


def mse_cost(y_pred, y_true):
    """Mean Square error cost function."""
    return 0.5*np.sum((y_pred - y_true)**2)  # /y.shape[0]


def mse_cost_derivative(y_pred, y_true):
    """Mean Square error cost function derivative."""
    return (y_pred - y_true)


def log_entropy(y_pred, y_true):
    """Cross entropy cost function."""
    return -np.sum(y_true * log(y))


def log_entropy_derivative(y_pred, y_true):
    """Derivative of cross entropy cost function."""
    # y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # return (y_pred - y_true)


def exponential_cost(y, y_true, tau=0.1):
    """Exponential cost function."""
    return tau*np.exp(1/tau * np.sum((y-y_true)**2))


def exponential_cost_derivative(y, y_true, tau=0.1):
    """Exponential cost function gradient."""
    return 2/tau * (y-y_true)*exponential_cost(y, y_true, tau)



# =============================================================================
# REGULARIZATIONS
# =============================================================================


def _l1(weights):
    """The L1 norm."""
    return np.linalg.norm(weights, ord=1)


def _l1_derivative(weights):
    """The derivative of the L1 norm."""
    # NOTE: Include this in report
    # https://math.stackexchange.com/questions/141101/minimizing-l-1-regularization
    return np.sign(weights)


def _l2(weights):
    """The L2 norm."""
    return 0.5*np.dot(weights, weights)


def _l2_derivative(weights):
    """The derivative of the L2 norm."""
    # NOTE: Include this in report
    # https://math.stackexchange.com/questions/2792390/derivative-of-
    # euclidean-norm-l2-norm
    return weights

