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
from lib import logistic_regression

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


def task1c(sk=False):
    """Task c) of project 2."""

    training_size = 0.8
    fract = 0.1

    print("Logistic regression")

    data_path = "../datafiles/MehtaIsingData"
    input_data = read_t("all", data_path)

    labels_data = pickle.load(open(os.path.join(
        data_path, "Ising2DFM_reSample_L40_T=All_labels.pkl"), "rb"))

    print("Data shape: {} Bytes: {:.2f} MB".format(
        input_data.shape, input_data.nbytes / (1024*1024)))
    print("Data label shape: {} Bytes: {:.2f} MB".format(
        labels_data.shape, labels_data.nbytes / (1024*1024)))

    # divide data into ordered, critical and disordered
    # X_ordered=input_data[:70000,:]
    X_ordered = input_data[:int(np.floor(70000*fract)), :]
    # Y_ordered=labels_data[:70000]
    Y_ordered = labels_data[:int(np.floor(70000*fract))]

    # X_critical=input_data[70000:100000,:]
    # Y_critical=labels_data[70000:100000]

    # X_disordered=input_data[100000:,:]
    X_disordered = input_data[100000:int(np.floor(100000*(1 + fract))), :]
    # Y_disordered=labels_data[100000:]
    Y_disordered = labels_data[100000:int(np.floor(100000*(1 + fract)))]

    del input_data, labels_data

    # define training and test data sets
    X = np.concatenate((X_ordered, X_disordered))
    Y = np.concatenate((Y_ordered, Y_disordered))

    # pick random data points from ordered and disordered states
    # to create the training and test sets
    X_train, X_test, Y_train, Y_test = sk_modsel.train_test_split(
        X, Y, train_size=training_size)

    # full data set
    # X=np.concatenate((X_critical,X))
    # Y=np.concatenate((Y_critical,Y))

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print()
    print(X_train.shape[0], 'train samples')
    # print(X_critical.shape[0], 'critical samples')
    print(X_test.shape[0], 'test samples')

    # define regularisation parameter
    lmbdas = np.logspace(-5, 5, 11)

    # preallocate data
    train_accuracy = np.zeros(lmbdas.shape, np.float64)
    test_accuracy = np.zeros(lmbdas.shape, np.float64)
    critical_accuracy = np.zeros(lmbdas.shape, np.float64)

    train_accuracy_SGD = np.zeros(lmbdas.shape, np.float64)
    test_accuracy_SGD = np.zeros(lmbdas.shape, np.float64)
    critical_accuracy_SGD = np.zeros(lmbdas.shape, np.float64)

    # loop over regularisation strength
    for i, lmbda in enumerate(lmbdas):

        # define logistic regressor
        if sk:
            logreg = sk_model.LogisticRegression(
                C=1.0/lmbda, random_state=1, verbose=0, max_iter=1E3, tol=1E-5)
        else:
            logreg = logistic_regression.LogisticRegression(
                penalty="l1", lr=1.0, max_iter=1E3)
        # fit training data
        logreg.fit(cp.deepcopy(X_train), cp.deepcopy(Y_train.reshape(-1, 1)))

        # check accuracy
        train_accuracy[i] = logreg.score(X_train, Y_train)
        test_accuracy[i] = logreg.score(X_test, Y_test)
        # critical_accuracy[i]=logreg.score(X_critical,Y_critical)

        print('accuracy: train, test, critical')
        print('liblin: %0.4f, %0.4f, %0.4f' %
              (train_accuracy[i], test_accuracy[i], critical_accuracy[i]))

        # define SGD-based logistic regression
        logreg_SGD = sk_model.SGDClassifier(loss='log', penalty='l2',
                                            alpha=lmbda, max_iter=100,
                                            shuffle=True, random_state=1,
                                            learning_rate='optimal')

        # fit training data
        logreg_SGD.fit(X_train, Y_train)

        # check accuracy
        train_accuracy_SGD[i] = logreg_SGD.score(X_train, Y_train)
        test_accuracy_SGD[i] = logreg_SGD.score(X_test, Y_test)
        # critical_accuracy_SGD[i]=logreg_SGD.score(X_critical,Y_critical)

        print('SGD: %0.4f, %0.4f, %0.4f' % (
            train_accuracy_SGD[i],
            test_accuracy_SGD[i],
            critical_accuracy_SGD[i]))

        print('finished computing %i/11 iterations' % (i+1))


def main():
    # task1b()
    task1c()


if __name__ == '__main__':
    main()
