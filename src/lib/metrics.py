#!/usr/bin/env python
import numpy as np


__all__ = ["mse", "r2", "bias", "timing_function"]


def mse(y_excact, y_predict, axis=0):
    """Mean Square Error

    Uses numpy to calculate the mean square error.

    MSE = (1/n) sum^(N-1)_i=0 (y_i - y_test_i)^2

    Args:
        y_excact (ndarray): response/outcome/dependent variable,
            size (N_samples, 1)
        y_predict (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: mean square error
    """

    assert y_excact.shape == y_predict.shape

    return np.mean((y_excact - y_predict)**2, axis=axis)[0]


def r2(y_excact, y_predict, axis=None):
    """R^2 score

    Uses numpy to calculate the R^2 score.

    R^2 = 1 - sum(y - y_test)/sum(y - mean(y_test))

    Args:
        y_excact (ndarray): response/outcome/dependent variable,
            size (N_samples, 1)
        y_predict (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: R^2 score
    """


    mse_excact_pred = np.sum((y_excact - y_predict)**2, axis=axis)
    variance_excact = np.sum((y_excact - np.mean(y_excact))**2)
    return (1.0 - mse_excact_pred/variance_excact)


def bias(y_excact, y_predict, axis=0):
    """Bias^2 of a excact y and a predicted y

    Args:
        y_excact (ndarray): response/outcome/dependent variable,
            size (N_samples, 1)
        y_predict (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: Bias^2
    """
    return np.mean((y_predict - np.mean(y_excact, keepdims=True, axis=axis))**2)


def ridge_regression_variance(X, sigma2, lmb):
    """Analytical variance for beta coefs in Ridge regression,
    from section 1.4.2, page 10, https://arxiv.org/pdf/1509.09169.pdf"""
    XT_X = X.T @ X
    W_lmb = XT_X + lmb * np.eye(XT_X.shape[0])
    W_lmb_inv = np.linalg.inv(W_lmb)
    return np.diag(sigma2 * W_lmb_inv @ XT_X @ W_lmb_inv.T)



def timing_function(func):
    """Time function decorator."""
    import time
    def wrapper(*args, **kwargs):
        t1 = time.clock()
        
        val = func(*args, **kwargs)

        t2 = time.clock()

        time_used = t2-t1
        
        print ("Time used with function {:s}: {:.10f} secs/ "
            "{:.10f} minutes".format(func.__name__, time_used, time_used/60.))
        
        return val

    return wrapper
