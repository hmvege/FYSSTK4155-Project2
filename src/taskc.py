#!/usr/bin/env python3

import numpy as np
import copy as cp
import os
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib import ising_1d as ising
from lib import regression as reg
from lib import metrics
from lib import bootstrap as bs
from lib import cross_validation as cv
from lib import logistic_regression as logreg

import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils

from tqdm import tqdm

from task_tools import read_t, load_pickle, save_pickle, print_parameters


def task1c(sk=False, figure_folder="../fig"):
    """Task c) of project 2."""
    print("="*80)
    print("Task c: Logistic regression classification")

    training_size = 0.8
    data_percentage = 0.01
    # options: float, inverse
    learning_rate = "inverse"
    max_iter = int(1e3)
    tolerance = 1e-6
    momentum = 0.0
    mini_batch_size = 20
    data_path = "../datafiles/MehtaIsingData"

    # Available solvers:
    # ["lr-gd", "gd", "cg", "sga", "sga-mb", "nr", "newton-cg"]
    solver = "lr-gd"
    # Available activation functions: sigmoid, softmax
    activation = "sigmoid"
    # Available penalites: l1, l2, elastic_net
    penalty = "elastic_net"

    # Define regularisation parameter
    lmbdas = np.logspace(-5, 5, 11)

    # data_ordered = []  # Defined as data less than T/J=2.0
    # data_critical = []  # Defined as data between T/J=2.0 and T/J=2.5
    # data_disordered = []  # Defined as data greater than T/J=2.5

    # for T in np.arange(0.25, 4.25, 0.25):
    #     fpath = os.path.join(data_path,
    #                          "Ising2DFM_reSample_L40_T={0:.2f}.pkl".format(T))

    #     print("Loads data for T={0:2.2f} from {1}".format(T, fpath))
    #     with open(fpath, "rb") as f:
    #         if T < 2.0:
    #             data_ordered.append(pickle.load(f))
    #         elif 2.0 <= T <= 2.5:
    #             data_critical.append(pickle.load(f))
    #         else:
    #             data_disordered.append(pickle.load(f))

    # data_ordered = np.asarray(data_ordered)
    # data_critical = np.asarray(data_critical)
    # data_disordered = np.asarray(data_disordered)

    input_data = read_t("all", data_path)

    labels_data = pickle.load(open(os.path.join(
        data_path, "Ising2DFM_reSample_L40_T=All_labels.pkl"), "rb"))

    print("Data shape: {} Bytes: {:.2f} MB".format(
        input_data.shape, input_data.nbytes / (1024*1024)))
    print("Data label shape: {} Bytes: {:.2f} MB".format(
        labels_data.shape, labels_data.nbytes / (1024*1024)))

    # Checks that a good data percentage has been provided
    assert (0 < data_percentage <= 1.0), ("bad data_percentage: "
                                          "{}".format(data_percentage))

    # divide data into ordered, critical and disordered, as is done in Metha
    # X_ordered = input_data[:70000, :]
    # Y_ordered = labels_data[:70000]
    # X_critical = input_data[70000:100000, :]
    # Y_critical = labels_data[70000:100000]
    # X_disordered = input_data[100000:, :]
    # Y_disordered = labels_data[100000:]

    X_ordered = input_data[:int(np.floor(70000*data_percentage)), :]
    Y_ordered = labels_data[:int(np.floor(70000*data_percentage))]

    # X_critical = input_data[70000:int(np.floor(100000*data_percentage)), :]
    # Y_critical = labels_data[70000:int(np.floor(100000*data_percentage))]

    X_disordered = input_data[100000:int(
        np.floor(100000*(1 + data_percentage))), :]
    Y_disordered = labels_data[100000:int(
        np.floor(100000*(1 + data_percentage)))]

    del input_data, labels_data

    # define training and test data sets
    X = np.concatenate((X_ordered, X_disordered))
    Y = np.concatenate((Y_ordered, Y_disordered))

    # pick random data points from ordered and disordered states
    # to create the training and test sets
    X_train, X_test, y_train, y_test = sk_modsel.train_test_split(
        X, Y, test_size=1-training_size)

    # full data set
    # X=np.concatenate((X_critical,X))
    # Y=np.concatenate((Y_critical,Y))

    print('X_train shape(samples, input):', X_train.shape)
    print('y_train shape(samples, input):', y_train.shape)

    run_sk_comparison(X_train, X_test, y_train, y_test,
                      lmbdas=lmbdas,
                      penalty=penalty,
                      activation=activation,
                      solver=solver,
                      learning_rate=learning_rate,
                      momentum=momentum,
                      mini_batch_size=mini_batch_size,
                      max_iter=max_iter,
                      tolerance=tolerance)


def run_lambda_penalty(X_train, X_test, y_train, y_test, lmbdas=lmbdas,
                       penalties=penalties, **kwargs):
    pass


def run_lambda_solver(X_train, X_test, y_train, y_test, lmbdas=lmbdas,
                      solvers=solvers, **kwargs):
    pass


def run_lambda_learning_rate_comparison(X_train, X_test, y_train, y_test,
                                        lmbdas=lmbdas,
                                        learning_rates=learning_rates,
                                        **kwargs):
    pass


def run_lambda_momentum(X_train, X_test, y_train, y_test, lmbdas=lmbdas,
                        momentums=momentums, **kwargs):
    pass


def run_sk_comparison(X_train, X_test, y_train, y_test,
                      lmbdas=lmbdas,
                      penalty=penalty,
                      activation=activation,
                      solver=solver,
                      learning_rate=learning_rate,
                      momentum=momentum,
                      mini_batch_size=mini_batch_size,
                      max_iter=max_iter,
                      tolerance=tolerance):
    """Runs a comparison between sk learn and our method."""

    print_parameters({"lambda": lambdas,
                      "sklearn": True,
                      "penalty": penalty,
                      "activation": activation,
                      "solver": solver,
                      "learning_rate": learning_rate,
                      "momentum": momentum,
                      "mini_batch_size": mini_batch_size,
                      "max_iter": max_iter,
                      "tolerance": tolerance})

    pickle_fname = ("accuracy_sklearn_penaltyelastic_net_actsigmoid"
                    "_solverlr-gd_lrinverse_mom0.0_tol1e-06.pkl")

    if os.path.isfile(pickle_fname):
        res_ = load_pickle(pickle_fname)
    else:
        res_ = logreg_core(X_train, X_test, y_train, y_test,
                           use_sk_learn=True,
                           lmbdas=lmbdas,
                           penalty=penalty,
                           activation=activation,
                           solver=solver,
                           learning_rate=learning_rate,
                           momentum=momentum,
                           mini_batch_size=mini_batch_size,
                           max_iter=max_iter,
                           tolerance=tolerance,
                           store_pickle=True,
                           verbose=True)

    # Retrieves results
    train_accuracy, test_accuracy, critical_accuracy, \
        train_accuracy_SK, test_accuracy_SK, critical_accuracy_SK, \
        train_accuracy_SGD, test_accuracy_SGD, critical_accuracy_SGD = res_

    print('Mean accuracy: train, test')

    print("HomeMade: {0:0.4f} +/- {1:0.2f}, {2:0.4f} +/- {3:0.2f}".format(
        np.mean(train_accuracy), np.std(train_accuracy),
        np.mean(test_accuracy), np.std(test_accuracy)))

    print("SK: {0:0.4f} +/- {1:0.2f}, {2:0.4f} +/- {3:0.2f}".format(
        np.mean(train_accuracy_SK), np.std(train_accuracy_SK),
        np.mean(test_accuracy_SK), np.std(test_accuracy_SK)))

    print("SGD: {0:0.4f} +/- {1:0.2f}, {2:0.4f} +/- {3:0.2f}".format(
        np.mean(train_accuracy_SGD), np.std(train_accuracy_SGD),
        np.mean(test_accuracy_SGD), np.std(test_accuracy_SGD)))

    plot_accuracy_comparison(lmbdas, train_accuracy, test_accuracy,
                             train_accuracy_SK, test_accuracy_SK,
                             train_accuracy_SGD, test_accuracy_SGD,
                             "logistic_accuracy_sklearn_comparison",
                             figure_folder)


def run_logreg_fit(X_train, X_test, y_train, y_test, use_sk_learn=False,
                   lmbdas=[None], penalties=[None],
                   activations=[None], solvers=[None],
                   learning_rates=[None],
                   momentums=[None], mini_batch_sizes=[None],
                   max_iter=None, tolerances=[None], store_pickle=False,
                   verbose=False):
    """Method for retrieveing data for given lists of hyperparameters

    Args:
        X_train (ndarray)
        X_test (ndarray)
        y_train (ndarray)
        y_test (ndarray)
        use_sk_learn (bool): if we are to use SK-learn. Default is False.
        lmbdas (list(float)): list of lmbdas.
        penalties (list(str): list of penalty types. Choices: l1, l2, 
            elastic_net.
        activations (list(str)): list of activation functions.
        solvers (list(str)): list of solver functions.
        learning_rates (list(str|float)): list of learning rates. 
            Options: float, inverse.
        momentums (list(float)): momentum strengths, list of floats.
        mini_batch_sizes (list(float)): list of minibatch sizes.
        max_iter (int): maximum number of iterations
        tolerances (list(floats)): list of tolerances
        store_pickle (bool): store results as pickles
        verbose (bool): more verbose output. Default is False.

    Returns:
        Dictionary with logreg accuracy scores and times
        SK-learn dictionary with accuracy scores and times
        SGD-SK-learn dictionary with accuracy scores and times
    """

    for penalty in penalties:
        for activation in activations:
            for solver in solvers:
                for lr in learning_rates:
                    for momentum in momentums:
                        for mb in mini_batch_sizes:
                            for tol in tolerances:
                                logreg_core(X_train, X_test,
                                            y_train, y_test,
                                            use_sk_learn=use_sk_learn,
                                            lmbdas=lmbdas,
                                            penalty=penalty,
                                            activation=activation,
                                            learning_rate=lr,
                                            momentum=momentum,
                                            mini_batch_size=mb,
                                            max_iter=max_iter,
                                            tolerance=tol,
                                            store_pickle=store_pickle,
                                            verbose=False)


def logreg_core(X_train, X_test, y_train, y_test, use_sk_learn=False,
                lmbdas=[None], penalty=None,
                activation=None, solver=None, learning_rate=None,
                momentum=None, mini_batch_size=None, max_iter=None,
                tolerance=None, store_pickle=False, verbose=False):
    """Method for retrieveing data for given lists of hyperparameters

    Args:
        X_train (ndarray)
        X_test (ndarray)
        y_train (ndarray)
        y_test (ndarray)
        use_sk_learn (bool): if we are to use SK-learn. Default is False.
        lmbdas (float): list of lmbdas
        penalty (str): penalty type. Choices: l1, l2, elastic_net
        activation (str): activation function.
        solver(str): solver function.
        learning_rate (str|float): learning rate. Options: float, inverse
        momentum (float): momentum strength
        mini_batch_size (float): minibatch size
        tolerance (float): tolerance, at what point we cut off the parameter 
            search.
        verbose (bool): more verbose output. Default is False

    Returns:
        Dictionary with logreg accuracy scores and times
        SK-learn dictionary with accuracy scores and times
        SGD-SK-learn dictionary with accuracy scores and times
    """

    # Sets up data arrays
    train_accuracy = np.zeros(lmbdas.shape, np.float64)
    test_accuracy = np.zeros(lmbdas.shape, np.float64)
    critical_accuracy = np.zeros(lmbdas.shape, np.float64)

    if use_sk_learn:
        train_accuracy_SK = np.zeros(lmbdas.shape, np.float64)
        test_accuracy_SK = np.zeros(lmbdas.shape, np.float64)
        critical_accuracy_SK = np.zeros(lmbdas.shape, np.float64)

        train_accuracy_SGD = np.zeros(lmbdas.shape, np.float64)
        test_accuracy_SGD = np.zeros(lmbdas.shape, np.float64)
        critical_accuracy_SGD = np.zeros(lmbdas.shape, np.float64)

    # Loops over regularisation strength
    for i, lmbda in enumerate(lmbdas):

        if verbose:
            print("")
            print("="*80)
            print("Lambda = ", lmbda)

        if use_sk_learn:
            # Sets up lambda for SK learn logreg methods
            if lmbda == 0.0:
                sk_lmbda = 1.0 / 10000000
            else:
                sk_lmbda = 1.0 / lmbda

            # Set sk penalty
            if penalty == "elastic_net":
                sk_penalty = "l2"
            else:
                sk_penalty = penalty

            # Define SK-learn logistic regressor
            logreg_SK = sk_model.LogisticRegression(
                penalty=sk_penalty,
                fit_intercept=False, C=sk_lmbda,
                max_iter=max_iter, tol=tolerance)

            # SK learn fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logreg_SK.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
            if verbose:
                print("SK-learn method done")

            # Retrieve accuracy scores for SK-learn
            train_accuracy_SK[i] = logreg_SK.score(X_train, y_train)
            test_accuracy_SK[i] = logreg_SK.score(X_test, y_test)

            # Sets up learning rate for SK-learn SGD
            if learning_rate == "inverse":
                sk_learning_rate = "optimal"
                eta0 = 0.0
            else:
                sk_learning_rate = "constant"
                eta0 = learning_rate

            # Sets up regularisation for SK-learn SGD
            if penalty == "elastic_net":
                sk_penalty = "elasticnet"

            # define SGD-based logistic regression
            logreg_SGD = sk_model.SGDClassifier(loss="log",
                                                penalty=sk_penalty,
                                                alpha=lmbda, max_iter=max_iter,
                                                shuffle=True, random_state=1,
                                                eta0=eta0,
                                                learning_rate=sk_learning_rate)

            # fit training data
            logreg_SGD.fit(X_train, y_train)
            if verbose:
                print("SGA SK-learn method done")

            # check accuracy
            train_accuracy_SGD[i] = logreg_SGD.score(X_train, y_train)
            test_accuracy_SGD[i] = logreg_SGD.score(X_test, y_test)
            # critical_accuracy_SGD[i]=logreg_SGD.score(X_critical,Y_critical)

        # Our implementation of logistic regression
        log_reg = logreg.LogisticRegression(penalty=penalty, solver=solver,
                                            activation=activation,
                                            tol=tolerance,
                                            alpha=lmbda, momentum=momentum,
                                            mini_batch_size=mini_batch_size,
                                            max_iter=max_iter)

        # Fit training data
        log_reg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train),
                    eta=learning_rate)
        if verbose:
            print("HomeMade method done")

        # Accuracy score for our implementation
        train_accuracy[i] = log_reg.score(X_train, y_train)
        test_accuracy[i] = log_reg.score(X_test, y_test)
        # critical_accuracy[i]=log_reg.score(X_critical,Y_critical)

        # Prints result from single lambda run
        if verbose:
            print('Accuracy scores: train, test, critical')
            print('HomeMade: {0:0.4f}, {1:0.4f}, {2:0.4f}'.format(
                train_accuracy[i], test_accuracy[i], critical_accuracy[i]))

        if use_sk_learn and verbose:
            print('SK: {0:0.4f}, {1:0.4f}, {2:0.4f}'.format(
                train_accuracy_SK[i], test_accuracy_SK[i],
                critical_accuracy_SK[i]))

            print('SGD: %0.4f, %0.4f, %0.4f' % (
                train_accuracy_SGD[i],
                test_accuracy_SGD[i],
                critical_accuracy_SGD[i]))

        # Prints iteration values
        if verbose:
            print("Finished computing {}/11 iterations".format(i+1))

    if verbose:
        print("")

    results = [train_accuracy, test_accuracy, critical_accuracy,
               train_accuracy_SK, test_accuracy_SK, critical_accuracy_SK,
               train_accuracy_SGD, test_accuracy_SGD, critical_accuracy_SGD]

    if store_pickle:
        if use_sk_learn:
            sk_str = "sklearn_"
        else:
            sk_str = ""

        fname = ("accuracy_{}penalty{}_act{}_solver{}_lr{}_mom{}_tol{}."
                 "pkl".format(sk_str, penalty, activation, solver,
                              str(learning_rate), str(momentum),
                              str(tolerance)))
        save_pickle(fname, results)

    return results


def plot_accuracy_comparison(lmbdas, train_accuracy, test_accuracy,
                             train_accuracy_SK, test_accuracy_SK,
                             train_accuracy_SGD, test_accuracy_SGD,
                             figname, figure_folder):
    """plot accuracy against regularisation strength."""
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.semilogx(lmbdas, train_accuracy,
                 marker="o", ls=(0, (3, 1, 1, 1)),  # Densely dashdotted
                 color="#7570b3",
                 label='Implemented train')
    ax1.semilogx(lmbdas, test_accuracy,
                 marker="x", ls=(0, (3, 1, 1, 1)),  # Densely dashdotted
                 color="#7570b3",
                 label='Implemented test')

    ax1.semilogx(lmbdas, train_accuracy_SK,
                 marker="o", ls=(0, (5, 1)),  # Densely dashed
                 color="#1b9e77",
                 label='SK-Learn train')
    ax1.semilogx(lmbdas, test_accuracy_SK,
                 marker="x", ls=(0, (5, 1)),  # Densely dashed
                 color="#1b9e77",
                 label='SK-Learn test')

    ax1.semilogx(lmbdas, train_accuracy_SGD,
                 marker="o", ls=(0, (3, 5, 1, 5)),  # Dashdotted
                 color="#d95f02",
                 label='SK-Learn SGD train')
    ax1.semilogx(lmbdas, test_accuracy_SGD,
                 marker="x", ls=(0, (3, 5, 1, 5)),  # Dashdotted
                 color="#d95f02",
                 label='SK-Learn SGD test')

    ax1.set_xlabel(r'$\lambda$')
    ax1.set_ylabel(r'$\mathrm{Accuracy}$')

    ax1.grid(True)
    ax1.legend(fontsize=8)

    figure_path = os.path.join(figure_folder, "{}.pdf".format(figname))
    fig.savefig(figure_path)
    print("Figure saved at {}".format(figure_path))
    plt.close(fig)


if __name__ == '__main__':
    task1c()
