#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np

import sklearn.model_selection as sk_modsel

from lib import neuralnetwork as nn

from task_tools import load_pickle, save_pickle, print_parameters, \
    plot_accuracy_scores, retrieve_2d_ising_data, convert_output


def task1e(figure_path="../fig"):
    """Task e) of project 2.

    Task: train the NN with the cross entropy function and compare with 
    Logistic Regression results from c).
    """
    print("="*80)
    print("Task e: neural network classification")
    data_path = "../datafiles/MehtaIsingData"
    data_percentage = 0.3
    training_size = 0.8
    learning_rate = 1.0
    max_iter = int(1e3)
    verbose = True

    print("Logistic regression")

    X, y = retrieve_2d_ising_data(data_path, data_percentage)

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

    data_train_labels = np.asarray(
        [convert_output(l, output_layer_shape) for l in y_train])
    data_test_labels = np.asarray(
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
    default_penalty = "l2"
    default_activation = "sigmoid"
    default_output_activation = "softmax"
    default_cost_function = "log_loss"
    default_learning_rate = "inverse"
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

    # Hyper parameters to chose from
    activation = ["sigmoid", "identity", "relu", "tanh", "heaviside"]
    output_activation = ["sigmoid", "identity", "softmax"]
    cost_function = ["mse", "log_loss", "exponential_cost"]
    learning_rates = np.logspace(-6, -1, 6)
    learning_rate = ["inverse"]
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

    print_parameters(**default_input_dict)
    nn_core(X_train, X_test, y_train, y_test, layers, **default_input_dict)


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


def nn_core(X_train, X_test, y_train, y_test,
            layers, lmbda=None, penalty=None,
            activation=None, output_activation=None,
            cost_function=None,
            learning_rate=None,
            weight_init=None,
            epochs=None,
            mini_batch_size=None, max_iter=None,
            tolerance=None, verbose=False):
    """Method for retrieveing data for a given set of hyperparameters

    Args:
        X_train (ndarray)
        X_test (ndarray)
        y_train (ndarray)
        y_test (ndarray)
        layers (list(int)): list of layer sizes
        lmbdas (float): list of lmbdas
        penalty (str): penalty type. Choices: l1, l2, elastic_net
        activation (str): activation function.
        output_activation (str): output activation function. Choices: 
            sigmoidal, softmax, identity.
        learning_rate (str|float): learning rate. Options: float, inverse
        mini_batch_size (float): minibatch size
        tolerance (float): tolerance, at what point we cut off the parameter 
            search.
        verbose (bool): more verbose output. Default is False

    Returns:
        Dictionary with logreg accuracy scores and times
        SK-learn dictionary with accuracy scores and times
        SGD-SK-learn dictionary with accuracy scores and times
    """

    if verbose:
        print("")
        print("="*80)
        print("Lambda = ", lmbda)

    # Our implementation of logistic regression
    # Sets up my MLP.
    MLP = nn.MultilayerPerceptron(layers,
                                  activation=activation,
                                  cost_function=cost_function,
                                  output_activation=output_activation,
                                  weight_init=weight_init,
                                  alpha=lmbda)

    MLP.train(X_train, y_train,
              data_test=X_test,
              data_test_labels=y_test,
              mini_batch_size=mini_batch_size,
              epochs=epochs,
              eta=learning_rate)

    # Accuracy score for our implementation
    train_accuracy = MLP.score(X_train, y_train)
    test_accuracy = MLP.score(X_test, y_test)

    train_accuracy_epochs = MLP.evaluate(X_train, y_train)
    test_accuracy_epochs = MLP.evaluate(X_test, y_test)
    # critical_accuracy[i]=log_reg.score(X_critical,Y_critical)

    # Prints result from single lambda run
    if verbose:
        print('Accuracy scores: train, test')
        print('MultilayerPerceptron: {0:0.4f}, {1:0.4f}'.format(
            train_accuracy, test_accuracy))

    return train_accuracy, test_accuracy, train_accuracy_epochs, \
        test_accuracy_epochs


if __name__ == '__main__':
    task1e()
