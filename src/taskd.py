#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import sklearn.model_selection as sk_modsel

from lib import ising_1d as ising

from task_tools import load_pickle, save_pickle, print_parameters, \
    plot_accuracy_scores, convert_output, nn_core, plot_heatmap


def task1d(figure_folder="../fig"):
    """Task d) of project 2.

    Task: train the NN and compare with Linear Regression results from b).
    """
    print("="*80)
    print("Task d: neural network regression")
    training_size = 0.5
    L_system_size = 20
    N_samples = 8000

    states, energies = ising.generate_1d_ising_data(L_system_size, N_samples)
    states = states.reshape((*states.shape, 1))

    X_train, X_test, y_train, y_test = \
        sk_modsel.train_test_split(states, energies, test_size=1-training_size,
                                   shuffle=False)

    input_layer_shape = X_train.shape[1]
    output_layer_shape = y_train.shape[1]

    print("Training size: ", X_train.shape[0])
    print("Test size:     ", X_test.shape[0])
    # exit(1)

    # Default hyperparameters
    default_activation = "sigmoid"
    default_output_activation = "identity"
    default_cost_function = "mse"
    default_learning_rate = 0.01
    default_eta0 = float(1e-5)
    default_mini_batch_size = 20
    default_hidden_layer_size = 10
    default_weight_init = "default"
    default_regularization = "l2"
    default_lambda_value = 0.01
    default_epochs = 200
    default_layers = [input_layer_shape, output_layer_shape]

    lambda_values = np.logspace(-4, 5, 10)
    lambda_values = [0.01]
    verbose = True

    default_input_dict = {
        "lmbda": default_lambda_value,
        "learning_rate": default_learning_rate,
        "eta0": default_eta0,
        "regularization": default_regularization,
        "cost_function": default_cost_function,
        "activation": default_activation,
        "output_activation": default_output_activation,
        "mini_batch_size": default_mini_batch_size,
        "weight_init": default_weight_init,
        "epochs": default_epochs,
        "verbose": verbose,
    }

    linreg_layers = [input_layer_shape, output_layer_shape]

    linreg_dict = default_input_dict.copy()
    linreg_dict["lmbda"] = 0.0

    # First find OLS comparison
    # Fit with 0 hidden layers and identity output, then store weights-matrix
    OLS_results = nn_core(X_train, X_test, y_train, y_test, linreg_layers,
                          return_weights=True, **linreg_dict)


    J_OLS = np.asarray(OLS_results[-3]).reshape((L_system_size, L_system_size))

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
        J_ridge = np.asarray(
            ridge_results[-1][-3]).reshape((L_system_size, L_system_size))
        J_lasso = np.asarray(
            lasso_results[-1][-3]).reshape((L_system_size, L_system_size))

        plot_heatmap(J_OLS, J_ridge, J_lasso, L_system_size, lmbda,
                     figure_folder,
                     "mlp_ising_1d_heatmap_lambda{}.pdf".format(lmbda))


if __name__ == '__main__':
    task1d()
