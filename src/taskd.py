#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np

import sklearn.model_selection as sk_modsel

from lib import ising_1d as ising

from task_tools import load_pickle, save_pickle, print_parameters, \
    plot_accuracy_scores, convert_output, nn_core


def task1d(figure_path="../fig"):
    """Task d) of project 2.

    Task: train the NN and compare with Linear Regression results from b).
    """
    print("="*80)
    print("Task d: neural network regression")
    training_size = 0.8
    L_system_size = 20
    N_samples = 1000
    fract = 0.01
    learning_rate = 1.0
    max_iter = int(1e3)
    tolerance = 1e-8
    verbose = True

    states, energies = ising.generate_1d_ising_data(L_system_size, N_samples)
    states = states.reshape((*states.shape, 1))

    X_train, X_test, y_train, y_test = \
        sk_modsel.train_test_split(states, energies, test_size=1-training_size,
                                   shuffle=False)
    input_layer_shape = X_train.shape[1]
    output_layer_shape = y_train.shape[1]

    # # One-liners <3
    # output_classes = set(list(map(lambda i: int(i[0]), energies.tolist())))
    # output_classes = sorted(list(output_classes))
    # output_layer_shape = len(output_classes)

    # y_train_labels = np.zeros((y_train.shape[0], output_layer_shape))
    # y_test_labels = np.zeros((y_test.shape[0], output_layer_shape))

    # for i_train in range(y_train_labels.shape[0]):
    #     y_train_labels[i_train][np.argmax(
    #         output_classes == y_train[i_train])] = 1

    # for i_test in range(y_test_labels.shape[0]):
    #     y_test_labels[i_test][np.argmax(output_classes == y_test[i_test])] = 1

    # data_train_labels = np.asarray(
    #     [convert_output(l[0], output_layer_shape) for l in y_train])
    # data_test_labels = np.asarray(
    #     [convert_output(l[0], output_layer_shape) for l in y_test])

    # Default hyperparameters
    default_penalty = "l2"
    default_activation = "sigmoid"
    default_output_activation = "identity"
    default_cost_function = "log_loss"
    default_learning_rate = "inverse"
    default_eta0 = 1.0
    default_mini_batch_size = 20
    default_hidden_layer_size = 10
    default_weight_init = "default"
    default_regularization = "l2"
    default_lambda_value = 1e-2
    default_epochs = 300
    default_layers = [input_layer_shape, output_layer_shape]

    # Hyper parameters to chose from
    activation = ["sigmoid", "identity", "relu", "tanh", "heaviside"]
    output_activation = ["sigmoid", "identity", "softmax"]
    cost_function = ["mse", "log_loss", "exponential_cost"]
    learning_rates = np.logspace(-6, -1, 6)
    learning_rate = "inverse"
    mini_batch_size = [10, 20, 30, 40, 50]
    layer_neurons = [5, 10, 15, 20, 25, 30, 40, 50]
    weight_init = ["default", "large"]
    lambda_values = np.logspace(-4, 5, 10)

    default_input_dict = {
        "lmbda": default_lambda_value,
        "learning_rate": default_learning_rate,
        "eta0": default_eta0,
        "regularization": default_regularization,
        "cost_function": default_cost_function,
        "penalty": default_penalty,
        "activation": default_activation,
        "output_activation": default_output_activation,
        "mini_batch_size": default_mini_batch_size,
        "weight_init": default_weight_init,
        "epochs": default_epochs,
        "max_iter": max_iter,
        "verbose": verbose,
    }

    linreg_layers = [input_layer_shape, output_layer_shape]

    linreg_dict = default_input_dict.copy()
    linreg_dict.pop("lmbda")
    linreg_dict["lmbda"] = 0.0

    # First find OLS comparison
    # Fit with 0 hidden layers and identity output, then store weights-matrix
    OLS_results = nn_core(X_train, X_test, y_train, y_test, linreg_layers,
                          return_weights=True, **linreg_dict)

    ridge_results = []
    lasso_results = []

    # Loop over alphas, then
    for lmbda in lambda_values:
        print("Lambda:", lmbda)

        # 1. Fit with L1
        linreg_dict = default_input_dict.copy()
        linreg_dict.pop("lmbda")
        linreg_dict["lmbda"] = lmbda
        linreg_dict["regularization"] = "l1"

        _res = nn_core(X_train, X_test, y_train, y_test,
                       linreg_layers, return_weights=True,
                       **linreg_dict)
        ridge_results.append(_res)

        # 2. Fit with L2
        linreg_dict = default_input_dict.copy()
        linreg_dict.pop("lmbda")
        linreg_dict["lmbda"] = lmbda
        linreg_dict["regularization"] = "l2"

        _res = nn_core(X_train, X_test, y_train, y_test,
                       linreg_layers, return_weights=True,
                       **linreg_dict)
        lasso_results.append(_res)

        # Plot stuff here as in b)


if __name__ == '__main__':
    task1d()
