#!/usr/bin/env python3
import numpy as np
import copy as cp
from tqdm import tqdm

import lib.metrics as metrics

import sklearn.model_selection as sk_modsel
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils


def sk_learn_k_fold_cv(x, y, z, kf_reg, design_matrix, k_splits=4,
                       test_percent=0.4, print_results=True):
    """Scikit Learn method for cross validation."""
    x_train, x_test, y_train, y_test = sk_modsel.train_test_split(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        test_size=test_percent, shuffle=True)
    kf = sk_modsel.KFold(n_splits=k_splits)

    X_test = design_matrix(x_test)
    X_train = design_matrix(x_train)

    y_pred_list = []
    beta_coefs = []

    for train_index, test_index in tqdm(kf.split(X_train), 
        desc="SciKit-Learn k-fold Cross Validation"):

        kX_train, kX_test = X_train[train_index], X_train[test_index]
        kY_train, kY_test = y_train[train_index], y_train[test_index]

        kf_reg.fit(kX_train, kY_train)
        y_pred_list.append(kf_reg.predict(X_test))

        beta_coefs.append(kf_reg.coef_)

    y_pred_list = np.asarray(y_pred_list)

    # Mean Square Error, mean((y - y_approx)**2)
    _mse = (y_test - y_pred_list)**2
    MSE = np.mean(np.mean(_mse, axis=0, keepdims=True))

    # Bias, (y - mean(y_approx))^2
    _mean_pred = np.mean(y_pred_list, axis=0, keepdims=True)
    bias = np.mean((y_test - _mean_pred)**2)

    # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
    R2 = np.mean(metrics.R2(y_test, y_pred_list, axis=0))

    # Variance, var(y_predictions)
    var = np.mean(np.var(y_pred_list, axis=0, keepdims=True))

    beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
    beta_coefs = np.asarray(beta_coefs).mean(axis=0)

    if print_results:
        print("R2:    {:-20.16f}".format(R2))
        print("MSE:   {:-20.16f}".format(MSE))
        print("Bias^2:{:-20.16f}".format(bias))
        print("Var(y):{:-20.16f}".format(var))
        print("Beta coefs: {}".format(beta_coefs))
        print("Beta coefs variances: {}".format(beta_coefs_var))
        print("Diff: {}".format(abs(MSE - bias - var)))

    results = {
            "y_pred": np.mean(y_pred_list, axis=0),
            "y_pred_var": np.var(y_pred_list, axis=0),
            "mse": MSE,
            "r2": R2,
            "var": var,
            "bias": bias,
            "beta_coefs": beta_coefs,
            "beta_coefs_var": beta_coefs_var,
            "beta_95c": np.sqrt(beta_coefs_var)*2,
            "diff": abs(MSE - bias - var),
        }

    return results


def sk_learn_bootstrap(x, y, z, design_matrix, kf_reg, N_bs=100,
                       test_percent=0.4, print_results=True):
    """Sci-kit learn bootstrap method."""

    x_train, x_test, y_train, y_test = sk_modsel.train_test_split(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        test_size=test_percent, shuffle=False)

    # Ensures we are on axis shape (N_observations, N_predictors)
    y_test = y_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    y_pred = np.empty((y_test.shape[0], N_bs))

    X_test = design_matrix(x_test)

    R2_ = np.empty(N_bs)
    mse_ = np.empty(N_bs)
    bias2_ = np.empty(N_bs)

    beta_coefs = []

    X_train = design_matrix(x_train)

    for i_bs in tqdm(range(N_bs), desc="SciKit-Learn bootstrap"):
        x_boot, y_boot = sk_utils.resample(X_train, y_train)
        # x_boot, y_boot = sk_utils.resample(x_train, y_train)
        # X_boot = design_matrix(x_boot)

        kf_reg.fit(X_boot, y_boot)
        # y_pred[:, i_bs] = kf_reg.predict(cp.deepcopy(x_test)).ravel()

        y_predict = kf_reg.predict(X_test)
        # print(sk_metrics.r2_score(y_test.flatten(), y_pred[:,i_bs].flatten()))

        # R2_[i_bs] = sk_metrics.r2_score(y_test.flatten(), y_pred[:,i_bs].flatten())
        # R2_[i_bs] = metrics.R2(y_test, y_predict)
        # mse_[i_bs] = metrics.mse(y_test.flatten(), y_pred[:, i_bs].flatten())
        # bias2_[i_bs] = metrics.bias2(
        #     y_test.flatten(), y_pred[:, i_bs].flatten())

        y_pred[:, i_bs] = y_predict.ravel()

        beta_coefs.append(kf_reg.coef_)

    # R2 = np.mean(R2_)
    # # print("R2 from each bs step = ",R2)
    # # # MSE = mse_.mean()
    # # # bias = bias2_.mean()
    # # R2 = np.mean(R2_list)

    # # R2 = (1 - np.sum(np.average((y_test - y_pred)**2, axis=1)) /
    # #       np.sum((y_test - np.average(y_test)**2)))
    # # print(R2)
    # print(y_test.shape, y_pred.shape)
    # s1 = np.sum((np.mean((y_test - y_pred)**2, axis=1)))
    # s2 = np.sum((y_test - np.mean(y_test))**2)
    # print ("R2=",1 - s1/s2)
    # R2 = (1 - np.sum(np.mean((y_test - y_pred)**2, axis=0, keepdims=True),keepdims=True) /
    #       np.sum((y_test - np.mean(y_test, keepdims=True)**2,),keepdims=True))
    # print(R2.mean())
    # R2 = R2.mean()
    R2 = np.mean(metrics.R2(y_test, y_pred, axis=0))

    # Mean Square Error, mean((y - y_approx)**2)
    _mse = ((y_test - y_pred))**2
    MSE = np.mean(np.mean(_mse, axis=1, keepdims=True))

    # Bias, (y - mean(y_approx))^2
    _mean_pred = np.mean(y_pred, axis=1, keepdims=True)
    bias = np.mean((y_test - _mean_pred)**2)

    # Variance, var(y_predictions)
    var = np.mean(np.var(y_pred, axis=1, keepdims=True))

    beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
    beta_coefs = np.asarray(beta_coefs).mean(axis=0)

    # # R^2 score, 1 - sum((y-y_approx)**2)/sum((y-mean(y))**2)
    # y_pred_mean = np.mean(y_pred, axis=1)
    # _y_test = y_test.reshape(-1)
    # print ("R2:", metrics.R2(_y_test, y_pred_mean))

    # _s1 = np.sum(((y_test - y_pred))**2, axis=1, keepdims=True)
    # _s2 = np.sum((y_test - np.mean(y_test))**2)
    # print (_s1.mean(), _s2)

    # R2 = 1 - _s1.mean()/_s2
    # print(np.array([sk_metrics.r2_score(y_test, y_pred[:,i]) for i in range(N_bs)]).mean())
    # R2 = metrics.R2(y_test, y_pred, axis=1)
    # R2 = np.mean(metrics.R2(y_test, y_pred, axis=1))
    # print(np.mean(metrics.R2(y_test, y_pred, axis=1)))
    # R2 = R2.mean()
    # print(R2.mean())

    if print_results:
        print("R2:    {:-20.16f}".format(R2))
        print("MSE:   {:-20.16f}".format(MSE))
        print("Bias^2:{:-20.16f}".format(bias))
        print("Var(y):{:-20.16f}".format(var))
        print("Beta coefs: {}".format(beta_coefs))
        print("Beta coefs variances: {}".format(beta_coefs_var))
        print("Diff: {}".format(abs(MSE - bias - var)))

    results = {
            "y_pred": np.mean(y_pred, axis=1),
            "y_pred_var": np.var(y_pred, axis=1),
            "mse": MSE,
            "r2": R2,
            "var": var,
            "bias": bias,
            "beta_coefs": beta_coefs,
            "beta_coefs_var": beta_coefs_var,
            "beta_95c": np.sqrt(beta_coefs_var)*2,
            "diff": abs(MSE - bias - var),
        }

    return results
