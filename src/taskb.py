#!/usr/bin/env python3

import numpy as np
import copy as cp
import os
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib import regression as reg
from lib import metrics
from lib import bootstrap as bs
from lib import cross_validation as cv
from lib import logistic_regression
from lib.ising_1d import generate_1d_ising_data

import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils

from task_tools import load_pickle

def task1b(pickle_fname, N_samples=1000, training_size=0.1, N_bs=200,
           L_system_size=20, figure_folder="../fig"):
    """Task b of project 2"""
    print("="*80)
    print("Task b")

    states, energies = generate_1d_ising_data(L_system_size, N_samples)

    X_train, X_test, y_train, y_test = \
        sk_modsel.train_test_split(states, energies, test_size=1-training_size,
                                   shuffle=False)

    lambda_values = np.logspace(-4, 5, 10)

    # y_pred_list (80, 10000)
    # y_test (80, 1)

    # Linear regression
    linreg = reg.OLSRegression()
    linreg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
    y_pred_linreg = linreg.predict(cp.deepcopy(X_test))
    y_pred_linreg_train = linreg.predict(cp.deepcopy(X_train))

    linreg_general_results = {
        "test": {"r2": metrics.r2(y_test, y_pred_linreg),
                 "mse": metrics.mse(y_test, y_pred_linreg),
                 "bias": metrics.bias(y_test, y_pred_linreg)},
        "train": {"r2": metrics.r2(y_train, y_pred_linreg_train),
                  "mse": metrics.mse(y_train, y_pred_linreg_train),
                  "bias": metrics.bias(y_train, y_pred_linreg_train)}}

    print("LINREG:")
    print("R2:  {:-20.16f}".format(linreg_general_results["test"]["r2"]))
    print("MSE: {:-20.16f}".format(linreg_general_results["test"]["mse"]))
    print("Bias: {:-20.16f}".format(linreg_general_results["test"]["bias"]))
    # print("Beta coefs: {}".format(linreg.coef_))
    # print("Beta coefs variances: {}".format(linreg.coef_var))

    J_leastsq = np.asarray(linreg.coef_).reshape((L_system_size, L_system_size))

    linreg_bs_results = bs.BootstrapWrapper(X_train, y_train,
                                            sk_model.LinearRegression(
                                                fit_intercept=False),
                                            N_bs, X_test=X_test,
                                            y_test=y_test)

    linreg_cvkf_results = cv.kFoldCVWrapper(X_train, y_train,
                                            sk_model.LinearRegression(
                                                fit_intercept=False), k=4,
                                            X_test=X_test, y_test=y_test)

    ridge_general_results = []
    ridge_bs_results = []
    ridge_cvkf_results = []

    lasso_general_results = []
    lasso_bs_results = []
    lasso_cvkf_results = []

    for lmbda in lambda_values:

        # Ridge regression
        ridge_reg = reg.RidgeRegression(lmbda)
        ridge_reg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
        y_pred_ridge = ridge_reg.predict(cp.deepcopy(X_test)).reshape(-1, 1)
        y_pred_ridge_train = ridge_reg.predict(
            cp.deepcopy(X_train)).reshape(-1, 1)
        ridge_general_results.append({
            "test": {
                "lambda": lmbda,
                "r2": metrics.r2(y_test, y_pred_ridge),
                "mse": metrics.mse(y_test, y_pred_ridge),
                "bias": metrics.bias(y_test, y_pred_ridge)},
            "train": {
                "lambda": lmbda,
                "r2": metrics.r2(y_train, y_pred_ridge_train),
                "mse": metrics.mse(y_train, y_pred_ridge_train),
                "bias": metrics.bias(y_train, y_pred_ridge_train)},
        })

        print("\nRIDGE (lambda={}):".format(lmbda))
        print("R2:  {:-20.16f}".format(
            ridge_general_results[-1]["test"]["r2"]))
        print("MSE: {:-20.16f}".format(
            ridge_general_results[-1]["test"]["mse"]))
        print("Bias: {:-20.16f}".format(
            ridge_general_results[-1]["test"]["bias"]))

        ridge_bs_results.append(
            bs.BootstrapWrapper(X_train, y_train,
                                reg.RidgeRegression(lmbda),
                                N_bs, X_test=X_test, y_test=y_test))

        ridge_cvkf_results.append(
            cv.kFoldCVWrapper(X_train, y_train,
                              reg.RidgeRegression(lmbda), k=4,
                              X_test=X_test, y_test=y_test))

        # Lasso regression
        lasso_reg = sk_model.Lasso(alpha=lmbda)

        # Filtering out annoing warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            lasso_reg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
            y_pred_lasso = lasso_reg.predict(
                cp.deepcopy(X_test)).reshape(-1, 1)
            y_pred_lasso_train = lasso_reg.predict(
                cp.deepcopy(X_train)).reshape(-1, 1)

        lasso_general_results.append({
            "test": {
                "lambda": lmbda,
                "r2": metrics.r2(y_test, y_pred_lasso),
                "mse": metrics.mse(y_test, y_pred_lasso),
                "bias": metrics.bias(y_test, y_pred_lasso)},
            "train": {
                "lambda": lmbda,
                "r2": metrics.r2(y_train, y_pred_lasso_train),
                "mse": metrics.mse(y_train, y_pred_lasso_train),
                "bias": metrics.bias(y_train, y_pred_lasso_train)},
        })

        print("\nLASSO (lambda={}):".format(lmbda))
        print("R2:  {:-20.16f}".format(
            lasso_general_results[-1]["test"]["r2"]))
        print("MSE: {:-20.16f}".format(
            lasso_general_results[-1]["test"]["mse"]))
        print("Bias: {:-20.16f}".format(
            lasso_general_results[-1]["test"]["bias"]))
        # print("Beta coefs: {}".format(lasso_reg.coef_))

        # Filtering out annoing warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            lasso_bs_results.append(
                bs.BootstrapWrapper(cp.deepcopy(X_train), cp.deepcopy(y_train),
                                    sk_model.Lasso(lmbda),
                                    N_bs, X_test=X_test, y_test=y_test))

            lasso_cvkf_results.append(
                cv.kFoldCVWrapper(cp.deepcopy(X_train), cp.deepcopy(y_train),
                                  sk_model.Lasso(lmbda), k=4,
                                  X_test=X_test, y_test=y_test))

        J_ridge = np.asarray(ridge_reg.coef_).reshape((L, L))
        J_lasso = np.asarray(lasso_reg.coef_).reshape((L, L))

        cmap_args = dict(vmin=-1., vmax=1., cmap='seismic')

        fig, axarr = plt.subplots(nrows=1, ncols=3)

        axarr[0].imshow(J_leastsq, **cmap_args)
        axarr[0].set_title(r'$\mathrm{OLS}$', fontsize=16)
        axarr[0].tick_params(labelsize=16)

        axarr[1].imshow(J_ridge, **cmap_args)
        axarr[1].set_title(
            r'$\mathrm{Ridge}, \lambda=%.4f$' % (lmbda), fontsize=16)
        axarr[1].tick_params(labelsize=16)

        im = axarr[2].imshow(J_lasso, **cmap_args)
        axarr[2].set_title(
            r'$\mathrm{LASSO}, \lambda=%.4f$' % (lmbda), fontsize=16)
        axarr[2].tick_params(labelsize=16)

        divider = make_axes_locatable(axarr[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25), fontsize=14)
        cbar.set_label(r'$J_{i,j}$', labelpad=-40,
                       y=1.12, fontsize=16, rotation=0)

        # plt.show()
        figure_path = os.path.join(
            figure_folder, "ising_1d_heatmap_lambda{}.pdf".format(lmbda))
        fig.savefig(figure_path)
        print("Figure for lambda={} stored at {}.".format(lmbda, figure_path))

        plt.close(fig)

    with open(pickle_fname, "wb") as f:
        pickle.dump({
            "ols": linreg_general_results,
            "ols_bs": linreg_bs_results,
            "ols_cv": linreg_cvkf_results,
            "ridge": ridge_general_results,
            "ridge_bs": ridge_bs_results,
            "ridge_cv": ridge_cvkf_results,
            "lasso": lasso_general_results,
            "lasso_bs": lasso_bs_results,
            "lasso_cv": lasso_cvkf_results}, f)
        print("Data pickled and dumped to: {:s}".format(pickle_fname))


def task1b_bias_variance_analysis(pickle_fname, figure_folder="../fig"):
    """Plot different bias/variance values"""
    lambda_values = np.logspace(-4, 5, 10)
    data = load_pickle(pickle_fname)

    def select_value(input_list, data_to_select, with_train=False):
        """Small function moving selected values to list."""
        if with_train:
            return {
                "train": [e["train"][data_to_select] for e in input_list],
                "test": [e["test"][data_to_select] for e in input_list]}
        else:
            return [e[data_to_select] for e in input_list]

    # OLS values
    ols_r2 = {"train": data["ols"]["train"]["r2"],
              "test": data["ols"]["test"]["r2"]}
    ols_mse = {"train": data["ols"]["train"]["mse"],
               "test": data["ols"]["test"]["mse"]}
    ols_bias = {"train": data["ols"]["train"]["bias"],
                "test": data["ols"]["test"]["bias"]}
    # Bootstrap OLS
    ols_bs_r2 = data["ols_bs"]["r2"]
    ols_bs_mse = data["ols_bs"]["mse"]
    ols_bs_bias = data["ols_bs"]["bias"]
    ols_bs_var = data["ols_bs"]["var"]
    # k-fold CV OLS
    ols_cv_r2 = data["ols_cv"]["r2"]
    ols_cv_mse = data["ols_cv"]["mse"]
    ols_cv_bias = data["ols_cv"]["bias"]
    ols_cv_var = data["ols_cv"]["var"]

    # General Ridge values
    ridge_r2 = select_value(data["ridge"], "r2", with_train=True)
    ridge_mse = select_value(data["ridge"], "mse", with_train=True)
    ridge_bias = select_value(data["ridge"], "bias", with_train=True)
    # Bootstrap Ridge values
    ridge_bs_mse = select_value(data["ridge_bs"], "mse")
    ridge_bs_bias = select_value(data["ridge_bs"], "bias")
    ridge_bs_var = select_value(data["ridge_bs"], "var")
    # k-fold CV Ridge values
    ridge_cv_mse = select_value(data["ridge_cv"], "mse")
    ridge_cv_bias = select_value(data["ridge_cv"], "bias")
    ridge_cv_var = select_value(data["ridge_cv"], "var")

    # General Lasso values
    lasso_r2 = select_value(data["lasso"], "r2", with_train=True)
    lasso_mse = select_value(data["lasso"], "mse", with_train=True)
    lasso_bias = select_value(data["lasso"], "bias", with_train=True)
    # Bootstrap Lasso
    lasso_bs_mse = select_value(data["lasso_bs"], "mse")
    lasso_bs_bias = select_value(data["lasso_bs"], "bias")
    lasso_bs_var = select_value(data["lasso_bs"], "var")
    # k-fold CV Lasso
    lasso_cv_mse = select_value(data["lasso_cv"], "mse")
    lasso_cv_bias = select_value(data["lasso_cv"], "bias")
    lasso_cv_var = select_value(data["lasso_cv"], "var")

    plot_dual_values(lambda_values, ridge_r2["test"],
                     lambda_values, lasso_r2["test"],
                     lambda_values, ols_r2["test"],
                     r"Ridge", r"Lasso", r"OLS", "ols_ridge_lasso_lambda_r2",
                     r"$\lambda$", r"$R^2$", figure_folder=figure_folder)
    plot_dual_values(lambda_values, ridge_mse["test"],
                     lambda_values, lasso_mse["test"],
                     lambda_values, ols_mse["test"],
                     r"Ridge", r"Lasso", r"OLS", "ols_ridge_lasso_lambda_mse",
                     r"$\lambda$", r"$\mathrm{MSE}$",
                     figure_folder=figure_folder)
    plot_dual_values(lambda_values, ridge_bias["test"],
                     lambda_values, lasso_bias["test"],
                     lambda_values, ols_bias["test"],
                     r"Ridge", r"Lasso", r"OLS", "ols_ridge_lasso_lambda_bias",
                     r"$\lambda$", r"$\mathrm{Bias}$",
                     figure_folder=figure_folder)

    # Plots Bootstrap analysis
    # plot_dual_values(lambda_values, ridge_bs_r2,
    #                  lambda_values, lasso_bs_r2,
    #                  lambda_values, ols_bs_r2,
    #                  r"Ridge", r"Lasso", r"OLS",
    #                  "ols_ridge_lasso_lambda_bs_r2",
    #                  r"$\lambda$", r"$R^2$", figure_folder=figure_folder)
    plot_dual_values(lambda_values, ridge_bs_mse,
                     lambda_values, lasso_bs_mse,
                     lambda_values, ols_bs_mse,
                     r"Ridge", r"Lasso", r"OLS",
                     "ols_ridge_lasso_lambda_bs_mse",
                     r"$\lambda$", r"$\mathrm{MSE}$",
                     figure_folder=figure_folder)
    plot_dual_values(lambda_values, ridge_bs_bias,
                     lambda_values, lasso_bs_bias,
                     lambda_values, ols_bs_bias,
                     r"Ridge", r"Lasso", r"OLS",
                     "ols_ridge_lasso_lambda_bs_bias",
                     r"$\lambda$", r"$\mathrm{Bias}$",
                     figure_folder=figure_folder)
    plot_dual_values(lambda_values, ridge_bs_var,
                     lambda_values, lasso_bs_var,
                     lambda_values, ols_bs_var,
                     r"Ridge", r"Lasso", r"OLS",
                     "ols_ridge_lasso_lambda_bs_var",
                     r"$\lambda$", r"$R^2$", figure_folder=figure_folder)

    # Plots Cross validation analysis
    # plot_dual_values(lambda_values, ridge_cv_r2,
    #                  lambda_values, lasso_cv_r2,
    #                  lambda_values, ols_cv_r2,
    #                  r"Ridge", r"Lasso", r"OLS",
    #                  "ols_ridge_lasso_lambda_cv_r2",
    #                  r"$\lambda$", r"$R^2$", figure_folder=figure_folder)
    plot_dual_values(lambda_values, ridge_cv_mse,
                     lambda_values, lasso_cv_mse,
                     lambda_values, ols_cv_mse,
                     r"Ridge", r"Lasso", r"OLS",
                     "ols_ridge_lasso_lambda_cv_mse",
                     r"$\lambda$", r"$\mathrm{MSE}$",
                     figure_folder=figure_folder)
    plot_dual_values(lambda_values, ridge_cv_bias,
                     lambda_values, lasso_cv_bias,
                     lambda_values, ols_cv_bias,
                     r"Ridge", r"Lasso", r"OLS",
                     "ols_ridge_lasso_lambda_cv_bias",
                     r"$\lambda$", r"$\mathrm{Bias}$",
                     figure_folder=figure_folder)
    plot_dual_values(lambda_values, ridge_cv_var,
                     lambda_values, lasso_cv_var,
                     lambda_values, ols_cv_var,
                     r"Ridge", r"Lasso", r"OLS",
                     "ols_ridge_lasso_lambda_cv_var",
                     r"$\lambda$", r"$R^2$", figure_folder=figure_folder)

    # Plots Bias-Variance for OLS
    plot_bias_variance(lambda_values, ols_bs_bias, ols_bs_var,
                       ols_bs_mse, "ols_bs_bias_variance_analysis",
                       figure_folder, x_hline=True)
    plot_bias_variance(lambda_values, ols_cv_bias, ols_cv_var,
                       ols_cv_mse, "ols_cv_bias_variance_analysis",
                       figure_folder, x_hline=True)

    # Plots Bias-Variance for Ridge
    plot_bias_variance(lambda_values, ridge_bs_bias, ridge_bs_var,
                       ridge_bs_mse, "ridge_bs_bias_variance_analysis",
                       figure_folder)
    plot_bias_variance(lambda_values, ridge_cv_bias, ridge_cv_var,
                       ridge_cv_mse, "ridge_cv_bias_variance_analysis",
                       figure_folder)

    # Plots Bias-Variance for Lasso
    plot_bias_variance(lambda_values, lasso_bs_bias, lasso_bs_var,
                       lasso_bs_mse, "lasso_bs_bias_variance_analysis",
                       figure_folder)
    plot_bias_variance(lambda_values, lasso_cv_bias, lasso_cv_var,
                       lasso_cv_mse, "lasso_cv_bias_variance_analysis",
                       figure_folder)

    # Plots R2 scores
    plot_all_r2(lambda_values, ols_r2["test"], ols_r2["train"],
                ridge_r2["test"], ridge_r2["train"], lasso_r2["test"],
                lasso_r2["train"], "r2_ols_ridge_lasso", figure_folder)


def plot_dual_values(x1, y1, x2, y2, x3, y3, label1, label2, label3,
                     figname, xlabel,
                     ylabel, figure_folder):
    """Plots two different values in a single window."""
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.semilogx(x1, y1, label=label1,
                 marker="o", ls=(0, (3, 1, 1, 1)),  # Densely dashdotted
                 color="#1b9e77")  # For Ridge
    ax1.semilogx(x2, y2, label=label2,
                 marker="x", ls=(0, (5, 1)),  # Densely dashed
                 color="#d95f02")  # For Lasso
    ax1.axhline(y3, label=label3,
                ls=(0, (5, 5)),  # Dashed
                color="#7570b3")  # For OLS

    # ax1.semilogx(x3, y3, label=label3)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)
    ax1.set_xlim(x1[0], x1[-1])
    ax1.legend()
    ax1.grid(True)

    figure_path = os.path.join(figure_folder, "{}.pdf".format(figname))
    fig.savefig(figure_path)
    print("Figure saved at {}".format(figure_path))
    plt.close(fig)

def plot_bias_variance(x, bias, variance, mse, figname, figure_folder,
                       x_hline=False):
    """Plots the bias-variance."""
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    if x_hline:
        ax1.axhline(bias, label=r"Bias$^2$",
                    marker="o", ls=(0, (3, 1, 1, 1)),  # Densely dashdotted
                    color="#1b9e77")
        ax1.axhline(variance, label=r"Var",
                    marker="x", ls=(0, (5, 1)),  # Densely dashed
                    color="#d95f02")
        ax1.axhline(mse, label=r"MSE",
                    ls=(0, (5, 5)),  # Dashed
                    color="#7570b3")
        ax1.set_xlim(x[0], x[-1])
    else:
        ax1.semilogx(x, bias, label=r"Bias$^2$",
                     marker="o", ls=(0, (3, 1, 1, 1)),  # Densely dashdotted
                     color="#1b9e77")
        ax1.semilogx(x, variance, label=r"Var",
                     marker="x", ls=(0, (5, 1)),  # Densely dashed
                     color="#d95f02")
        ax1.semilogx(x, mse, label=r"MSE",
                     ls=(0, (5, 5)),  # Dashed
                     color="#7570b3")
        ax1.set_xlim(x[0], x[-1])

    ax1.set_xlabel(r"$\lambda$")
    ax1.legend()
    ax1.grid(True)

    figure_path = os.path.join(figure_folder, "{}.pdf".format(figname))
    fig.savefig(figure_path)
    print("Figure saved at {}".format(figure_path))
    plt.close(fig)    


def plot_all_r2(lmbda_values, r2_ols_test, r2_ols_train, r2_ridge_test,
                r2_ridge_train, r2_lasso_test, r2_lasso_train, figname,
                figure_folder):
    """Plots all r2 scores together."""

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    # OLS
    ax1.axhline(r2_ols_test, label=r"OLS test",
                 marker="o", ls=(0, (3, 1, 1, 1)),  # Densely dashdotted
                 color="#7570b3")
    ax1.axhline(r2_ols_train, label=r"OLS train",
                 marker="x", ls=(0, (3, 1, 1, 1)),  # Densely dashdotted
                 color="#7570b3")

    # Ridge
    ax1.semilogx(lmbda_values, r2_ridge_test, label=r"Ridge test",
                 marker="o", ls=(0, (5, 1)),  # Densely dashed
                 color="#1b9e77")
    ax1.semilogx(lmbda_values, r2_ridge_train, label=r"Ridge train",
                 marker="x", ls=(0, (5, 1)),  # Densely dashed
                 color="#1b9e77")

    # Lasso
    ax1.semilogx(lmbda_values, r2_lasso_test, label=r"Lasso test",
                 marker="o", ls=(0, (3, 5, 1, 5)),  # Dashdotted
                 color="#d95f02")
    ax1.semilogx(lmbda_values, r2_lasso_train, label=r"Lasso train",
                 marker="x", ls=(0, (3, 5, 1, 5)),  # Dashdotted
                 color="#d95f02")

    ax1.set_xlim(lmbda_values[0], lmbda_values[-1])
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel(r"$R^2$")
    ax1.legend()
    ax1.grid(True)

    figure_path = os.path.join(figure_folder, "{}.pdf".format(figname))
    fig.savefig(figure_path)
    print("Figure saved at {}".format(figure_path))

if __name__ == '__main__':
    pickle_fname = "bs_kf_data_1b.pkl"
    task1b(pickle_fname)
    task1b_bias_variance_analysis(pickle_fname)