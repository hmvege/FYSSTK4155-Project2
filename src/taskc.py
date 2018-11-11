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
from lib import logistic_regression

import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils

from task_tools import read_t


def task1c(sk=False, figure_folder="../fig"):
    """Task c) of project 2."""
    print("="*80)
    print("Task c: Logistic regression classification")

    training_size = 0.8
    fract = 0.01
    learning_rate = 1.0
    max_iter = int(1e3)
    tolerance = 1e-5
    data_path = "../datafiles/MehtaIsingData"

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

    figure_path = os.path.join(figure_folder, "accuracy.pdf")
    plt.savefig(figure_path)

    plt.show()


if __name__ == '__main__':
    task1c()
