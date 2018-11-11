#!/usr/bin/env python3

import numpy as np
import copy as cp
import os
import pickle
import sys
import warnings
import argparse

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

# # Proper LaTeX font
# import matplotlib as mpl
# mpl.rc("text", usetex=True)
# mpl.rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
# mpl.rcParams["font.family"] += ["serif"]


def read_t(t="all", root="."):
    """Loads an ising model data set."""
    if t == "all":
        data = pickle.load(open(os.path.join(
            root, "Ising2DFM_reSample_L40_T=All.pkl"), "rb"))
    else:
        data = pickle.load(open(os.path.join(
            root, "Ising2DFM_reSample_L40_T=%.2f.pkl".format(t)), "rb"))

    return np.unpackbits(data).astype(int).reshape(-1, 1600)


def task1b(pickle_fname, N_samples=1000, training_size=0.1, N_bs=200,
           L_system_size=20, figure_folder="../fig"):
    """Task b of project 2"""
    np.random.seed(1234)

    # system size
    L = L_system_size

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

    linreg_general_results = {
        "r2": metrics.r2(y_test, y_pred_linreg),
        "mse": metrics.mse(y_test, y_pred_linreg),
        "bias": metrics.bias(y_test, y_pred_linreg)}

    print("LINREG:")
    print("R2:  {:-20.16f}".format(linreg_general_results["r2"]))
    print("MSE: {:-20.16f}".format(linreg_general_results["mse"]))
    print("Bias: {:-20.16f}".format(linreg_general_results["bias"]))
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
        ridge_general_results.append({
            "lambda": lmbda,
            "r2": metrics.r2(y_test, y_pred_ridge),
            "mse": metrics.mse(y_test, y_pred_ridge),
            "bias": metrics.bias(y_test, y_pred_ridge),
        })

        print("\nRIDGE (lambda={}):".format(lmbda))
        print("R2:  {:-20.16f}".format(ridge_general_results[-1]["r2"]))
        print("MSE: {:-20.16f}".format(ridge_general_results[-1]["mse"]))
        print("Bias: {:-20.16f}".format(ridge_general_results[-1]["bias"]))
        # print("Beta coefs: {}".format(ridge_reg.coef_))
        # print("Beta coefs variances: {}".format(ridge_reg.coef_var))

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

        lasso_general_results.append({
            "lambda": lmbda,
            "r2": metrics.r2(y_test, y_pred_lasso),
            "mse": metrics.mse(y_test, y_pred_lasso),
            "bias": metrics.bias(y_test, y_pred_lasso),
        })

        print("\nLASSO (lambda={}):".format(lmbda))
        print("R2:  {:-20.16f}".format(lasso_general_results[-1]["r2"]))
        print("MSE: {:-20.16f}".format(lasso_general_results[-1]["mse"]))
        print("Bias: {:-20.16f}".format(lasso_general_results[-1]["bias"]))
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

        plt.close(fig)

    with open(pickle_fname, "wb") as f:
        pickle.dump([linreg_general_results, linreg_bs_results,
                     linreg_cvkf_results, ridge_general_results,
                     ridge_bs_results, ridge_cvkf_results,
                     lasso_general_results, lasso_bs_results,
                     lasso_cvkf_results], f)
        print("Data pickled and dumped to: {:s}".format(pickle_fname))


def load_pickle(picke_file_name):
    with open(picke_file_name, "rb") as f:
        data = pickle.load(f)
        print("Pickle file loaded: {}".format(picke_file_name))
    return data


def task1b_bias_variance_analysis(pickle_fname, figure_folder="../fig"):
    """Plot different bias/variance values"""
    lambda_values = np.logspace(-4, 5, 10)
    data = load_pickle(pickle_fname)

    def select_value(input_list, data_to_select):
        """Small function moving selected values to list."""
        return [e[data_to_select] for e in input_list]

    # OLS values
    ols_r2 = data[0]["r2"]
    # General Ridge values
    ridge_r2 = select_value(data[3], "r2")
    ridge_mse = select_value(data[3], "mse")
    ridge_bias = select_value(data[3], "bias")
    # Bootstrap Ridge values
    ridge_bs_mse = select_value(data[4], "mse")
    ridge_bs_bias = select_value(data[4], "bias")
    ridge_bs_var = select_value(data[4], "var")
    # k-fold CV Ridge values
    ridge_kfcv_mse = select_value(data[5], "mse")
    ridge_kfcv_bias = select_value(data[5], "bias")
    ridge_kfcv_var = select_value(data[5], "var")
    # General Lasso values
    lasso_r2 = select_value(data[6], "r2")
    lasso_mse = select_value(data[6], "mse")
    lasso_bias = select_value(data[6], "bias")
    # Bootstrap Lasso
    lasso_bs_mse = select_value(data[7], "mse")
    lasso_bs_bias = select_value(data[7], "bias")
    lasso_bs_var = select_value(data[7], "var")
    # k-fold CV Lasso
    lasso_kfcv_mse = select_value(data[8], "mse")
    lasso_kfcv_bias = select_value(data[8], "bias")
    lasso_kfcv_var = select_value(data[8], "var")

    plot_dual_values(lambda_values, ridge_r2, lambda_values, lasso_r2,
                     r"Ridge", r"Lasso", "ridge_lasso_lambda_r2",
                     r"$\lambda$", r"$R^2$", figure_folder=figure_folder)
    plot_dual_values(lambda_values, ridge_mse, lambda_values, lasso_mse,
                     r"Ridge", r"Lasso", "ridge_lasso_lambda_mse",
                     r"$\lambda$", r"$\mathrm{MSE}$")
    plot_dual_values(lambda_values, ridge_bias, lambda_values, lasso_bias,
                     r"Ridge", r"Lasso", "ridge_lasso_lambda_bias",
                     r"$\lambda$", r"$\mathrm{Bias}$")


def plot_dual_values(x1, y1, x2, y2, label1, label2, figname, xlabel,
                     ylabel, figure_folder):
    """Plots two different values in a single window."""
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.semilogx(x1, y1, label=label1)
    ax1.semilogx(x2, y2, label=label2)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)
    ax1.legend()

    figure_path = os.path.join(figure_folder, "{}.pdf".format(figname))
    fig.savefig(figure_path)
    print("Figure saved at {}".format(figure_path))
    plt.show()


def task1c(sk=False, figure_folder="../fig"):
    """Task c) of project 2."""
    print("="*80)
    print("Logistic regression")

    training_size = 0.8
    fract = 0.01
    learning_rate = 1.0
    max_iter = int(1e3)
    tolerance = 1e-5
    data_path = "../datafiles/MehtaIsingData"

    data_ordered = []  # Defined as data less than T/J=2.0
    data_critical = []  # Defined as data between T/J=2.0 and T/J=2.5
    data_disordered = []  # Defined as data greater than T/J=2.5

    for T in np.arange(0.25, 4.25, 0.25):
        fpath = os.path.join(data_path,
                             "Ising2DFM_reSample_L40_T={0:.2f}.pkl".format(T))

        print("Loads data for T={0:2.2f} from {1}".format(T, fpath))
        with open(fpath, "rb") as f:
            if T < 2.0:
                data_ordered.append(pickle.load(f))
            elif 2.0 <= T <= 2.5:
                data_critical.append(pickle.load(f))
            else:
                data_disordered.append(pickle.load(f))

    data_ordered = np.asarray(data_ordered)
    data_critical = np.asarray(data_critical)
    data_disordered = np.asarray(data_disordered)

    input_data = read_t("all", data_path)

    labels_data = pickle.load(open(os.path.join(
        data_path, "Ising2DFM_reSample_L40_T=All_labels.pkl"), "rb"))

    print("Data shape: {} Bytes: {:.2f} MB".format(
        input_data.shape, input_data.nbytes / (1024*1024)))
    print("Data label shape: {} Bytes: {:.2f} MB".format(
        labels_data.shape, labels_data.nbytes / (1024*1024)))

    # divide data into ordered, critical and disordered, as is done in Metha
    X_ordered = input_data[:70000, :]
    # X_ordered = input_data[:int(np.floor(70000*fract)), :]
    Y_ordered = labels_data[:70000]
    # Y_ordered = labels_data[:int(np.floor(70000*fract))]

    X_critical = input_data[70000:100000, :]
    Y_critical = labels_data[70000:100000]

    X_disordered = input_data[100000:, :]
    # X_disordered = input_data[100000:int(np.floor(100000*(1 + fract))), :]
    Y_disordered = labels_data[100000:]
    # Y_disordered = labels_data[100000:int(np.floor(100000*(1 + fract)))]

    del input_data, labels_data

    # define training and test data sets
    X = np.concatenate((X_ordered, X_disordered))
    Y = np.concatenate((Y_ordered, Y_disordered))

    # pick random data points from ordered and disordered states
    # to create the training and test sets
    X_train, X_test, Y_train, Y_test = sk_modsel.train_test_split(
        X, Y, test_size=1-training_size)

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

    train_accuracy_SK = np.zeros(lmbdas.shape, np.float64)
    test_accuracy_SK = np.zeros(lmbdas.shape, np.float64)
    critical_accuracy_SK = np.zeros(lmbdas.shape, np.float64)

    train_accuracy_SGD = np.zeros(lmbdas.shape, np.float64)
    test_accuracy_SGD = np.zeros(lmbdas.shape, np.float64)
    critical_accuracy_SGD = np.zeros(lmbdas.shape, np.float64)

    # loop over regularisation strength
    for i, lmbda in enumerate(lmbdas):
        print("lambda = ", lmbda)

        # define logistic regressor
        logreg_SK = sk_model.LogisticRegression(
            fit_intercept=False, C=1.0/lmbda,
            max_iter=max_iter, tol=tolerance)

        logreg = logistic_regression.LogisticRegression(
            solver="lr-gd", activation="sigmoid", penalty="l2",
            tol=tolerance, max_iter=max_iter, alpha=lmbda)

        # fit training data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logreg_SK.fit(cp.deepcopy(X_train), cp.deepcopy(Y_train))
        print("SK-learn method done")

        logreg.fit(cp.deepcopy(X_train), cp.deepcopy(Y_train))
        print("Manual method done")

        # check accuracy
        train_accuracy_SK[i] = logreg_SK.score(X_train, Y_train)
        test_accuracy_SK[i] = logreg_SK.score(X_test, Y_test)

        train_accuracy[i] = logreg.score(X_train, Y_train)
        test_accuracy[i] = logreg.score(X_test, Y_test)
        # critical_accuracy[i]=logreg.score(X_critical,Y_critical)

        print('accuracy: train, test, critical')
        print('HomeMade: {0:0.4f}, {1:0.4f}, {2:0.4f}'.format(
            train_accuracy[i], test_accuracy[i], critical_accuracy[i]))

        print('SK: {0:0.4f}, {1:0.4f}, {2:0.4f}'.format(
            train_accuracy_SK[i], test_accuracy_SK[i],
            critical_accuracy_SK[i]))

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

    print('mean accuracy: train, test')
    print(r'HomeMade: %0.4f +/- %0.2f, %0.4f +/- %0.2f' % (
        np.mean(train_accuracy),
        np.std(train_accuracy),
        np.mean(test_accuracy),
        np.std(test_accuracy)))

    print('SK: %0.4f +/- %0.2f, %0.4f +/- %0.2f' % (
        np.mean(train_accuracy_SK),
        np.std(train_accuracy_SK),
        np.mean(test_accuracy_SK),
        np.std(test_accuracy_SK)))

    print('SGD: %0.4f +/- %0.2f, %0.4f +/- %0.2f' % (
        np.mean(train_accuracy_SGD),
        np.std(train_accuracy_SGD),
        np.mean(test_accuracy_SGD),
        np.std(test_accuracy_SGD)))

    # plot accuracy against regularisation strength
    plt.semilogx(lmbdas, train_accuracy, '.-b', label='HomeMade train')
    plt.semilogx(lmbdas, test_accuracy, '.-r', label='HomeMade test')

    plt.semilogx(lmbdas, train_accuracy_SK, '<--g', label='SK train')
    plt.semilogx(lmbdas, test_accuracy_SK, '<--b', label='SK test')

    plt.semilogx(lmbdas, train_accuracy_SGD, '*:r', label='SGD train')
    plt.semilogx(lmbdas, test_accuracy_SGD, '*:g', label='SGD test')

    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\mathrm{accuracy}$')

    plt.grid()
    plt.legend()

    figure_path = os.path.join(figure_folder, "accuracy.png")
    plt.savefig(figure_path)

    plt.show()


def task1d(figure_path="../fig"):
    """Task d) of project 2.

    Task: train the NN and compare with Linear Regression results from b).
    """
    training_size = 0.8
    fract = 0.01
    learning_rate = 1.0
    max_iter = int(1e3)
    tolerance = 1e-8

    print("Logistic regression")

    data_path = "../datafiles/MehtaIsingData"
    input_data = read_t("all", data_path)

    labels_data = pickle.load(open(os.path.join(
        data_path, "Ising2DFM_reSample_L40_T=All_labels.pkl"), "rb"))

    print("Come back later")
    sys.exit()


def task1e(figure_path="../fig"):
    """Task e) of project 2.

    Task: train the NN with the cross entropy function and compare with 
    Logistic Regression results from c).
    """
    training_size = 0.8
    fract = 0.01
    learning_rate = 1.0
    max_iter = int(1e3)
    tolerance = 1e-8

    print("Logistic regression")

    data_path = "../datafiles/MehtaIsingData"
    input_data = read_t("all", data_path)

    labels_data = pickle.load(open(os.path.join(
        data_path, "Ising2DFM_reSample_L40_T=All_labels.pkl"), "rb"))


def main():
    # Initiating parsers
    prog_desc = ("FYS-STK4155 Project 2 command-line utility for using "
                 "Machine Learning on Ising models data.")
    parser = argparse.ArgumentParser(prog="Project 2 ML Analyser",
                                     description=(prog_desc))
    parser.add_argument(
        "-figf", "--figure_folder", default="../fig", type=str,
        help="output path for figures.")

    # Sets up some subparser for each task.
    subparser = parser.add_subparsers(dest="subparser")

    # Task b
    taskb_parser = subparser.add_parser(
        "b", help=("Runs task b: finds the coupling constant for 1d Ising, "
                   "using Linear, Ridge and Lasso regression"))
    taskb_parser.add_argument("-pk", "--pickle_filename",
                              default="bs_kf_data_1b.pkl",
                              type=str,
                              help=("Filename for storing task b analysis "
                                    "output."))
    taskb_parser.add_argument("-N", "-N_samples", default=1000, type=int,
                              help="N 1D Ising samples to generate")

    # N_samples=1000, training_size=0.1, N_bs=200,
    #        L_system_size=20

    # taskb_parser.add_argument(
    #     "-figp", "--figure_path", default="../fig", type=str,
    #     help="output path for figures.")

    # Task c
    taskc_parser = subparser.add_parser(
        "c", help=("Runs task c: finds the phase of Ising matrices at "
                   "different temperatures using logisitc regression"))
    taskc_parser.add_argument("-pk", "--pickle-filename")
    # taskc_parser.add_argument(
    #     "-figp", "--figure_path", default="../fig", type=str,
    #     help="output path for figures.")

    # Task d
    taskd_parser = subparser.add_parser(
        "d", help=("Runs task d: uses a neural net to perform the regression"
                   " from b"))
    taskd_parser.add_argument("-pk", "--pickle-filename")
    # taskd_parser.add_argument(
    #     "-figp", "--figure_path", default="../fig", type=str,
    #     help="output path for figures.")

    # Task e
    taske_parser = subparser.add_parser(
        "e", help=("Runs task e: uses a neural net to perform the "
                   "classification from c"))
    taske_parser.add_argument("-pk", "--pickle-filename")
    # taske_parser.add_argument(
    #     "-figp", "--figure_path", default="../fig", type=str,
    #     help="output path for figures.")

    args = parser.parse_args()
    # if len(sys.argv) < 2:
    #     args = parser.parse_args(["b"])
    # else:
    #     args = parser.parse_args()

    if args.subparser == "b":
        task1b(args.pickle_filename, figure_folder=args.figure_folder)
        task1b_bias_variance_analysis(
            args.pickle_filename, figure_folder=args.figure_folder)
    if args.subparser == "c":
        task1c(figure_folder=args.figure_folder)
    if args.subparser == "d":
        task1d(figure_folder=args.figure_folder)
    if args.subparser == "e":
        task1e(figure_folder=args.figure_folder)


if __name__ == '__main__':
    main()
