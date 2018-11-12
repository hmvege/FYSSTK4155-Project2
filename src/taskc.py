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
from lib import logistic_regression as logreg

import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils

from tqdm import tqdm

from task_tools import read_t, load_pickle, save_pickle, print_parameters, \
    plot_accuracy_scores, retrieve_2d_ising_data


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
    verbose=True

    # Available solvers:
    # ["lr-gd", "gd", "cg", "sga", "sga-mb", "nr", "newton-cg"]
    solver = "lr-gd"
    # Available activation functions: sigmoid, softmax
    activation = "sigmoid"
    # Available penalites: l1, l2, elastic_net
    penalty = "elastic_net"

    # Define regularisation parameter
    lmbdas = np.logspace(-5, 5, 11)

    X, y = retrieve_2d_ising_data(data_path, data_percentage)

    # pick random data points from ordered and disordered states
    # to create the training and test sets
    X_train, X_test, y_train, y_test = sk_modsel.train_test_split(
        X, y, test_size=1-training_size)

    # full data set
    # X=np.concatenate((X_critical,X))
    # Y=np.concatenate((Y_critical,Y))

    print('X_train shape(samples, input):', X_train.shape)
    print('y_train shape(samples, input):', y_train.shape)

    run_sk_comparison(X_train, X_test, y_train, y_test,
                      lmbdas=lmbdas,
                      penalty="l2",
                      activation=activation,
                      solver=solver,
                      learning_rate=learning_rate,
                      momentum=momentum,
                      mini_batch_size=mini_batch_size,
                      max_iter=max_iter,
                      tolerance=tolerance,
                      verbose=verbose)

    run_lambda_penalty(X_train, X_test, y_train, y_test,
                       lmbdas=lmbdas,
                       use_sk_learn=False,
                       penalties=["l1", "l2", "elastic_net"],
                       activation=activation,
                       solver=solver,
                       learning_rate=learning_rate,
                       momentum=momentum,
                       mini_batch_size=mini_batch_size,
                       max_iter=max_iter,
                       tolerance=tolerance,
                       verbose=verbose)

    run_lambda_solver(X_train, X_test, y_train, y_test,
                      lmbdas=lmbdas,
                      penalty="l2",
                      activation=activation,
                      solvers=["lr-gd", "cg", "newton-cg"],
                      learning_rate="inverse",
                      momentum=momentum,
                      mini_batch_size=mini_batch_size,
                      max_iter=max_iter,
                      tolerance=tolerance,
                      verbose=verbose)

    run_lambda_learning_rate_comparison(X_train, X_test, y_train, y_test,
                                        lmbdas=lmbdas,
                                        penalty="l2",
                                        activation=activation,
                                        solver="lr-gd",
                                        learning_rates=["inverse", 0.001,
                                                        0.005, 0.01, 0.05,
                                                        0.1, 0.5, 1.0],
                                        momentum=momentum,
                                        mini_batch_size=mini_batch_size,
                                        max_iter=max_iter,
                                        tolerance=tolerance,
                                        verbose=verbose)

    run_lambda_momentum(X_train, X_test, y_train, y_test,
                        lmbdas=lmbdas,
                        penalty="l2",
                        activation=activation,
                        solver="lr-gd",
                        learning_rate=learning_rate,
                        momentums=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                        mini_batch_size=mini_batch_size,
                        max_iter=max_iter,
                        tolerance=tolerance,
                        verbose=verbose)


def run_lambda_penalty(X_train, X_test, y_train, y_test,
                       lmbdas=None,
                       penalties=None,
                       figure_folder="../fig",
                       **kwargs):

    param_dict = {"lmbdas": lmbdas,
                  "sklearn": True,
                  "figure_folder": figure_folder,
                  "penalties": penalties}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    test_accuracy_values = []
    train_accuracy_values = []

    for penalty in penalties:
        print("Regularisation:", penalty)

        pickle_fname = ("lambda_penalty_accuracy_penalty{}_actsigmoid"
                        "_solver{}_lrinverse_mom0.0_tol1e-06."
                        "pkl".format(penalty, kwargs["solver"]))

        if os.path.isfile(pickle_fname):
            res_ = load_pickle(pickle_fname)
        else:
            res_ = logreg_core(X_train, X_test, y_train, y_test,
                               lmbdas=lmbdas,
                               penalty=penalty,
                               store_pickle=True,
                               pickle_fname=pickle_fname,
                               **kwargs)

        train_accuracy, test_accuracy, critical_accuracy = res_

        test_accuracy_values.append(test_accuracy)
        train_accuracy_values.append(train_accuracy)

    plot_accuracy_scores(lmbdas, train_accuracy_values, test_accuracy_values,
                         [r"$L^1$", r"$L^2$", r"Elastic net"],
                         "accuracy_regularisation_scores", r"$\lambda$",
                         r"Accuracy")


def run_lambda_solver(X_train, X_test, y_train, y_test,
                      lmbdas=None,
                      solvers=None,
                      figure_folder="../fig",
                      **kwargs):

    param_dict = {"lmbdas": lmbdas,
                  "sklearn": True,
                  "figure_folder": figure_folder,
                  "solver": solvers}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    test_accuracy_values = []
    train_accuracy_values = []

    for solver in solvers:
        print("Solver:", solver)

        pickle_fname = ("lambda_solver_accuracy_penalty{}_actsigmoid"
                        "_solver{}_lr{}_mom0.0_tol1e-06."
                        "pkl".format(kwargs["penalty"], solver,
                                     kwargs["learning_rate"]))

        if os.path.isfile(pickle_fname):
            res_ = load_pickle(pickle_fname)
        else:
            res_ = logreg_core(X_train, X_test, y_train, y_test,
                               use_sk_learn=False,
                               lmbdas=lmbdas,
                               solver=solver,
                               store_pickle=True,
                               pickle_fname=pickle_fname,
                               **kwargs)

        train_accuracy, test_accuracy, critical_accuracy = res_

        test_accuracy_values.append(test_accuracy)
        train_accuracy_values.append(train_accuracy)

    plot_accuracy_scores(lmbdas, train_accuracy_values, test_accuracy_values,
                         [r"Optimized Gradient Descent", r"Conjugate Gradient",
                          r"Newtons-CG"],
                         "accuracy_solver_scores", r"$\lambda$",
                         r"Accuracy")


def run_lambda_learning_rate_comparison(X_train, X_test, y_train, y_test,
                                        lmbdas=None,
                                        learning_rates=None,
                                        figure_folder="../fig",
                                        **kwargs):

    param_dict = {"lmbdas": lmbdas,
                  "sklearn": True,
                  "figure_folder": figure_folder,
                  "learning_rates": learning_rates}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    test_accuracy_values = []
    train_accuracy_values = []

    for lr in learning_rates:
        print ("Learning rate: ", lr)

        pickle_fname = ("lambda_lr_accuracy_penalty{}_actsigmoid"
                        "_solver{}_lr{}_mom0.0_tol1e-06."
                        "pkl".format(kwargs["solver"],
                                     kwargs["penalty"], str(lr)))

        if os.path.isfile(pickle_fname):
            res_ = load_pickle(pickle_fname)
        else:
            res_ = logreg_core(X_train, X_test, y_train, y_test,
                               use_sk_learn=False,
                               lmbdas=lmbdas,
                               learning_rate=lr,
                               store_pickle=True,
                               pickle_fname=pickle_fname,
                               **kwargs)

        train_accuracy, test_accuracy, critical_accuracy = res_

        test_accuracy_values.append(test_accuracy)
        train_accuracy_values.append(train_accuracy)

    lr_labels = [r"Optimized"]
    lr_labels += [r"$\eta={0:.2f}$".format(lr) for lr in learning_rates[1:]]
    plot_accuracy_scores(lmbdas, train_accuracy_values, test_accuracy_values,
                         lr_labels, "accuracy_learning_rate_scores",
                         r"$\lambda$", r"Accuracy")


def run_lambda_momentum(X_train, X_test, y_train, y_test,
                        lmbdas=None,
                        momentums=None,
                        figure_folder="../fig",
                        **kwargs):

    param_dict = {"lmbdas": lmbdas,
                  "sklearn": True,
                  "figure_folder": figure_folder,
                  "momentums": momentums}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    test_accuracy_values = []
    train_accuracy_values = []

    for momentum in momentums:
        print ("Momentum: ", momentum)
        pickle_fname = ("lambda_mom_accuracy_penalty{}_actsigmoid"
                        "_solverlr-gd_lrinverse_mom{}_tol1e-06."
                        "pkl".format(kwargs["penalty"], str(momentum)))

        if os.path.isfile(pickle_fname):
            res_ = load_pickle(pickle_fname)
        else:
            res_ = logreg_core(X_train, X_test, y_train, y_test,
                               use_sk_learn=False,
                               lmbdas=lmbdas,
                               momentum=momentum,
                               store_pickle=True,
                               pickle_fname=pickle_fname,
                               **kwargs)

        train_accuracy, test_accuracy, critical_accuracy = res_

        test_accuracy_values.append(test_accuracy)
        train_accuracy_values.append(train_accuracy)

    plot_accuracy_scores(lmbdas, train_accuracy_values, test_accuracy_values,
                         [r"\gamma={0:.1e}".format(m) for m in momentums],
                         "accuracy_momentum_scores", r"$\lambda$",
                         r"Accuracy")


def run_sk_comparison(X_train, X_test, y_train, y_test,
                      lmbdas=None,
                      penalty=None,
                      activation=None,
                      solver=None,
                      learning_rate=None,
                      momentum=None,
                      mini_batch_size=None,
                      max_iter=None,
                      tolerance=None,
                      verbose=False,
                      figure_folder="../fig"):
    """Runs a comparison between sk learn and our method."""

    param_dict = {"lmbdas": lmbdas,
                  "sklearn": True,
                  "penalty": penalty,
                  "activation": activation,
                  "solver": solver,
                  "learning_rate": learning_rate,
                  "momentum": momentum,
                  "mini_batch_size": mini_batch_size,
                  "max_iter": max_iter,
                  "tolerance": tolerance}
    print_parameters(**param_dict)

    pickle_fname = ("sk_comparison_accuracy_sklearn_penalty{}_actsigmoid"
                    "_solverlr-gd_lrinverse_mom0.0_tol1e-06"
                    ".pkl".format(penalty))

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
                           pickle_fname=pickle_fname,
                           verbose=verbose)

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
                tolerance=None, store_pickle=False, pickle_fname=None,
                verbose=False):
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
        store_pickle (bool): saves output as pickle
        pickle_fname (str): pickle filename
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

    results = [train_accuracy, test_accuracy, critical_accuracy]

    if use_sk_learn:
        results += [train_accuracy_SK, test_accuracy_SK,
                    critical_accuracy_SK, train_accuracy_SGD,
                    test_accuracy_SGD, critical_accuracy_SGD]

    if store_pickle:
        if use_sk_learn:
            sk_str = "sklearn_"
        else:
            sk_str = ""

        if isinstance(pickle_fname, type(None)):
            fname = ("accuracy_{}penalty{}_act{}_solver{}_lr{}_mom{}_tol{}."
                     "pkl".format(sk_str, penalty, activation, solver,
                                  str(learning_rate), str(momentum),
                                  str(tolerance)))
        else:
            fname = pickle_fname
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
