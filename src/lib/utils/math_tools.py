#!/usr/bin/env python3

import numpy as np
import abc
from scipy.special import expit

AVAILABLE_ACTIVATIONS = ["identity", "sigmoid", "relu", "tanh", "heaviside"]

AVAILABLE_OUTPUT_ACTIVATIONS = [
    "identity", "sigmoid", "softmax"]

AVAILABLE_COST_FUNCTIONS = ["mse", "log_loss", "exponential_cost",
                            "hellinger_distance",
                            "kullback_leibler_divergence",
                            "generalized_kullback_leibler_divergence",
                            "itakura_saito_distance"]

AVAILABLE_REGULARIZATIONS = ["l1", "l2", "elastic_net"]

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
    # print(x.shape, z_exp.shape)
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

class _BaseCost:
    """Base cost function class."""
    @staticmethod
    @abc.abstractmethod
    def __call__(a, y):
        """Returns cost function.

        Args:
            a (ndarray): layer output.
            y (ndarray): true output.
        Return:
            (float): cost function output.
        """
        return None

    @staticmethod
    @abc.abstractmethod
    def delta(a, y, x):
        return None


class MSECost(_BaseCost):
    @staticmethod
    def __call__(a, y):
        """Returns cost function.

        Args:
            a (ndarray): all layer outputs.
            y (ndarray): all true outputs.
        Return:
            (float): cost function output.
        """
        return 0.5*np.mean(np.linalg.norm(a - y, axis=1)**2, axis=0)

    @staticmethod
    def delta(a, y, x):
        return (a - y) * x


class LogEntropyCost(_BaseCost):
    @staticmethod
    def __call__(a, y):
        """Returns cost function.

        Args:
            a (ndarray): layer output.
            y (ndarray): true output.
        Return:
            (float): cost function output.
        """
        return - np.mean(y*np.log(a) + (1 - y)*np.log(1 - a))

    @staticmethod
    def delta(a, y, x):
        return (a - y.reshape(a.shape))


class ExponentialCost(_BaseCost):
    """Exponential cost function."""
    @staticmethod
    def cost(a, y, tau=0.1):
        """Returns cost function.

        Args:
            a (ndarray): layer output.
            y (ndarray): true output.
        Return:
            (float): cost function output.
        """
        return tau*np.exp(1/tau * np.sum((y-y_true)**2))

    @staticmethod
    def delta(a, y, x, tua=0.1):
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


def _elastic_net(weights):
    """The elastic net regularization, L_en = L1 + L2."""
    return np.linalg.norm(weights, ord=1) + 0.5*np.dot(weights, weights)


def _elastic_net_derivative(weights):
    """Derivative of elastic net is just L1 and L2 derivatives combined."""
    return np.sign(weights) + weights


class _BaseRegularization:
    """Base cost function class."""
    @staticmethod
    @abc.abstractmethod
    def __call__(weights):
        """Returns the regularization."""
        return None

    @staticmethod
    @abc.abstractmethod
    def delta(weights):
        return None


class L1Regularization(_BaseRegularization):
    @staticmethod
    def __call__(weights):
        """Returns the regularization."""
        return np.linalg.norm(weights, ord=1)

    @staticmethod
    def derivative(weights):
        return np.sign(weights)


class L2Regularization(_BaseRegularization):
    @staticmethod
    def __call__(weights):
        """Returns the regularization."""
        return 0.5*np.dot(weights, weights)

    @staticmethod
    def derivative(weights):
        return weights


class ElasticNetRegularization(_BaseRegularization):
    @staticmethod
    def __call__(weights):
        """Returns the regularization."""
        return np.linalg.norm(weights, ord=1) + 0.5*np.dot(weights, weights)

    @staticmethod
    def derivative(weights):
        return np.sign(weights) + weights
