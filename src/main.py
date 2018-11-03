import numpy as np
import copy as cp
import os
import pickle
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn

from lib import ising_1d as ising
from lib import regression as reg
from lib import metrics
from lib import bootstrap as bs
from lib import cross_validation as cv

import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils


def read_t(t="all", root="."):
    """Loads an ising model data set."""
    if t == "all":
        data = pickle.load(open(os.path.join(
            root, "Ising2DFM_reSample_L40_T=All.pkl"), "rb"))
    else:
        data = pickle.load(open(os.path.join(
            root, "Ising2DFM_reSample_L40_T=%.2f.pkl".format(t)), "rb"))

    return np.unpackbits(data).astype(int).reshape(-1, 1600)


def task1b():
    """Task b of project 2"""

    # Number of samples to generate
    N_samples = 1000
    training_size = 0.1

    N_bs = 200

    np.random.seed(12)

    # system size
    L = 20

    # create 10000 random Ising states
    states = np.random.choice([-1, 1], size=(N_samples, L))

    # calculate Ising energies
    energies = ising.ising_energies(states, L)
    energies = energies.reshape((energies.shape[0], 1))

    # reshape Ising states into RL samples: S_iS_j --> X_p
    states = np.einsum('...i,...j->...ij', states, states)

    # Reshaping to correspond to energies.
    # Shamelessly stolen a lot of from:
    # https://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB_CVI-linreg_ising.html
    # E.g. why did no-one ever tell me about einsum?
    # That's awesome - no way I would have discovered that by myself.
    stat_shape = states.shape
    states = states.reshape((stat_shape[0], stat_shape[1]*stat_shape[2]))

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

    print("LINREG:")
    print("R2:  {:-20.16f}".format(metrics.r2(y_test, y_pred_linreg)))
    print("MSE: {:-20.16f}".format(metrics.mse(y_test, y_pred_linreg)))
    print("Bias: {:-20.16f}".format(metrics.bias(y_test, y_pred_linreg)))
    # print("Beta coefs: {}".format(linreg.coef_))
    # print("Beta coefs variances: {}".format(linreg.coef_var))

    J_leastsq = np.asarray(linreg.coef_).reshape((L, L))

    linreg_bs_results = bs.BootstrapWrapper(X_train, y_train,
                                            sk_model.LinearRegression(
                                                fit_intercept=False),
                                            N_bs, X_test=X_test,
                                            y_test=y_test)

    linreg_cvkf_results = cv.kFoldCVWrapper(X_train, y_train,
                                            sk_model.LinearRegression(
                                                fit_intercept=False), k=4,
                                            X_test=X_test, y_test=y_test)

    ridge_bs_results = []
    ridge_cvkf_results = []

    lasso_bs_results = []
    lasso_cvkf_results = []

    for lmbda in lambda_values:

        # Ridge regression
        ridge_reg = reg.RidgeRegression(lmbda)
        ridge_reg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
        y_pred_ridge = ridge_reg.predict(cp.deepcopy(X_test)).reshape(-1, 1)

        print("\nRIDGE:")
        print("R2:  {:-20.16f}".format(metrics.r2(y_test, y_pred_ridge)))
        print("MSE: {:-20.16f}".format(metrics.mse(y_test, y_pred_ridge)))
        print("Bias: {:-20.16f}".format(metrics.bias(y_test, y_pred_ridge)))
        # print("Beta coefs: {}".format(ridge_reg.coef_))
        # print("Beta coefs variances: {}".format(ridge_reg.coef_var))

        ridge_bs_results.append(
            bs.BootstrapWrapper(X_train, y_train,
                                reg.RidgeRegression(lmbda),
                                N_bs, X_test=X_test, y_test=y_test))

        ridge_cfkf_results.append(
            bs.BootstrapWrapper(X_train, y_train,
                                reg.RidgeRegression(lmbda),
                                N_bs, X_test=X_test, y_test=y_test))

        # Lasso regression
        lasso_reg = sk_model.Lasso(alpha=lmbda)
        lasso_reg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
        y_pred_lasso = lasso_reg.predict(cp.deepcopy(X_test)).reshape(-1, 1)

        print("\nLASSO:")
        print("R2:  {:-20.16f}".format(metrics.r2(y_test, y_pred_lasso)))
        print("MSE: {:-20.16f}".format(metrics.mse(y_test, y_pred_lasso)))
        print("Bias: {:-20.16f}".format(metrics.bias(y_test, y_pred_lasso)))
        # print("Beta coefs: {}".format(lasso_reg.coef_))

        lasso_bs_results.append(
            bs.BootstrapWrapper(X_train, y_train,
                                reg.LassoRegression(lmbda),
                                N_bs, X_test=X_test, y_test=y_test))

        lasso_cvkf_results.append(
            bs.BootstrapWrapper(X_train, y_train,
                                reg.LassoRegression(lmbda),
                                N_bs, X_test=X_test, y_test=y_test))

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
        fig.savefig("../fig/ising_1d_heatmap_lambda{}.pdf".format(lmbda))

        plt.close(fig)

    # Plot different bias/variance values


def task1c():
    """Task c) of project 2."""

    print("Logistic regression")

    data_path = "../datafiles/MehtaIsingData"
    input_data = read_t("all", data_path)

    labels_data = pickle.load(open(os.path.join(
            data_path, "Ising2DFM_reSample_L40_T=All_labels.pkl"), "rb"))

    print("Data shape: {} Bytes: {:.2f} MB".format(input_data.shape, input_data.nbytes / (1024*1024)))
    print("Data label shape: {} Bytes: {:.2f} MB".format(labels_data.shape, labels_data.nbytes / (1024*1024)))



def main():
    # task1b()
    task1c()

if __name__ == '__main__':
    main()
