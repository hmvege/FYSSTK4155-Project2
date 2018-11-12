#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np

import sklearn.model_selection as sk_modsel

from lib import neuralnetwork as nn

from task_tools import load_pickle, save_pickle, print_parameters, \
    plot_accuracy_scores, retrieve_2d_ising_data, convert_output, \
    nn_core


def task1e(figure_path="../fig"):
    """Task e) of project 2.

    Task: train the NN with the cross entropy function and compare with 
    Logistic Regression results from c).
    """
    print("="*80)
    print("Task e: neural network classification")
    data_path = "../datafiles/MehtaIsingData"
    data_size = 10000
    training_size = 0.8
    learning_rate = 1.0
    max_iter = int(1e3)
    verbose = True

    print("Neural Network classification")

    X, y = retrieve_2d_ising_data(data_path, data_size)

    # pick random data points from ordered and disordered states
    # to create the training and test sets
    X_train, X_test, y_train, y_test = sk_modsel.train_test_split(
        X, y, test_size=1-training_size)

    input_layer_shape = X_train.shape[-1]
    output_classes = set(list(map(lambda i: int(i), y_test)))
    output_classes = sorted(list(output_classes))
    output_layer_shape = len(output_classes)

    X_train = X_train.reshape((*X_train.shape, 1))
    X_test = X_test.reshape((*X_test.shape, 1))

    y_train = np.asarray(
        [convert_output(l, output_layer_shape) for l in y_train])
    y_test = np.asarray(
        [convert_output(l, output_layer_shape) for l in y_test])

    # data_train_labels = np.a

    # Hyper-parameters to test for:
    # Activation options: "sigmoid", "identity", "relu", "tanh", "heaviside"
    activation = "sigmoid"
    # Cost function options: "mse", "log_loss", "exponential_cost"
    cost_function = "log_loss"
    # Output activation options:  "identity", "sigmoid", "softmax"
    output_activation = "sigmoid"
    # Weight initialization options:
    # default(sigma=1/sqrt(N_samples)), large(sigma=1.0)
    weight_init = "default"
    alpha = 0.0
    mini_batch_size = 20
    epochs = 100
    eta = "inverse"  # Options: float, 'inverse'

    # Default hyperparameters
    default_activation = "sigmoid"
    default_output_activation = "softmax"
    default_cost_function = "log_loss"
    default_learning_rate = "inverse"
    default_eta0 = 0.1
    default_regularization = "l2"
    default_mini_batch_size = 20
    default_hidden_layer_size = 10
    default_weight_init = "default"
    default_lambda_value = 0.0
    default_epochs = 200
    default_layers = [input_layer_shape, default_hidden_layer_size,
                      output_layer_shape]

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

    # Hyper parameters to chose from
    activation = ["sigmoid", "identity", "relu", "tanh", "heaviside"]
    output_activation = ["sigmoid", "identity", "softmax"]
    cost_function = ["mse", "log_loss", "exponential_cost"]
    learning_rates = np.logspace(-6, -1, 6)
    mini_batch_size = [10, 20, 30, 40, 50]
    layer_neurons = [5, 10, 15, 20, 25, 30, 40, 50]
    weight_init = ["default", "large"]
    lambda_values = np.logspace(-4, 5, 10)

    lmbda_eta_params = default_input_dict.copy()
    lmbda_eta_params.pop("lmbda")
    lmbda_eta_params.pop("learning_rate")
    lmbda_eta_params["figure_folder"] = figure_path

    run_lambda_eta(X_train, X_test, y_train, y_test, default_layers,
                   lmbdas=lambda_values, learning_rates=learning_rates,
                   **lmbda_eta_params)


    # The following run produces near perfect accuracy
    nn_core(X_train, X_test, y_train, y_test, default_layers, **default_input_dict)


def run_lambda_mini_batches():
    pass


def run_lambda_neurons():
    pass


def run_neurons_eta():
    pass


def run_neurons_training_size():
    pass


def run_epoch_activations():
    pass


def run_epoch_output_activations():
    pass


def run_epoch_cost_functions():
    pass


def run_epoch_weight_init():
    pass


def run_lambda_eta(X_train, X_test, y_train, y_test, layers,
                   lmbdas=None, learning_rates=None,
                   figure_folder="../fig", **kwargs):

    param_dict = {"lmbdas": lmbdas, "learning_rates": learning_rates}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    # Double for-loop for all results
    pickle_fname = "mlp_lambda_eta_results.pkl"

    if os.path.isfile(pickle_fname):
        data = load_pickle(pickle_fname)
    else:
        data = {lmbda: {eta: {} for eta in learning_rates} for lmbda in lmbdas}
        for lmbda in lmbdas:
            for eta in learning_rates:
                print("Lambda: {} Eta: {}".format(lmbda, eta))
                res_ = nn_core(X_train, X_test, y_train, y_test, layers,
                               lmbda=lmbda, learning_rate=eta, **kwargs)
                data[lmbda][eta] = res_

        save_pickle(pickle_fname, data)

    plot_accuracy_scores(lmbdas, train_accuracy_values, test_accuracy_values,
                         [r"\gamma={0:.1e}".format(m) for m in momentums],
                         "accuracy_momentum_scores", r"$\lambda$",
                         r"Accuracy")


def nn_loop_wrapper(loop_arg, store_pickle, pickle_fname, *args, **kwargs):

    train_accuracy_values = []
    test_accuracy_values = []
    train_accuracy_epochs_values = []
    test_accuracy_epochs_values = []

    for hyperparam in kwargs[loop_arg]:
        res_ = nn_core(*args, loop_arg=hyperparam, **kwargs)

        train_accuracy_values.append(res_[0])
        test_accuracy_values.append(res_[1])
        train_accuracy_epochs_values.append(res_[2])
        test_accuracy_epochs_values.append(res_[3])

    results = [train_accuracy_values, test_accuracy_values,
               train_accuracy_epochs_values, test_accuracy_epochs_values]

    if store_pickle:
        fname = ("mlp_{}.pkl".format(pickle_fname))
        save_pickle(fname, results)
    return results




if __name__ == '__main__':
    task1e()
