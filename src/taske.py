#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np

import sklearn.model_selection as sk_modsel

from lib import neuralnetwork as nn

from task_tools import load_pickle, save_pickle, print_parameters, \
    plot_accuracy_scores, retrieve_2d_ising_data, convert_output, \
    nn_core, heatmap_plotter, plot_epoch_accuracy, convert_nn_core_to_dict

from matplotlib import rc, rcParams
rc("text", usetex=True)
rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
rcParams["font.family"] += ["serif"]


def task1e(figure_path="../fig"):
    """Task e) of project 2.

    Task: train the NN with the cross entropy function and compare with 
    Logistic Regression results from c).
    """
    print("="*80)
    print("Task e: neural network classification")
    data_path = "../datafiles/MehtaIsingData"
    data_size = 10000
    training_size = 0.5
    learning_rate = 1.0
    max_iter = int(1e3)
    verbose = False
    try_get_pickle=False

    print("Neural Network classification")

    [input_layer_shape, output_layer_shape, X_train, X_test, y_train,
        y_test] = \
        retrieve_2d_data_formatted(data_path, data_size, training_size)

    # X, y = retrieve_2d_ising_data(data_path, data_size)

    # # pick random data points from ordered and disordered states
    # # to create the training and test sets
    # X_train, X_test, y_train, y_test = sk_modsel.train_test_split(
    #     X, y, test_size=1-training_size)

    # input_layer_shape = X_train.shape[-1]
    # output_classes = set(list(map(lambda i: int(i), y_test)))
    # output_classes = sorted(list(output_classes))
    # output_layer_shape = len(output_classes)

    # X_train = X_train.reshape((*X_train.shape, 1))
    # X_test = X_test.reshape((*X_test.shape, 1))

    # y_train = np.asarray(
    #     [convert_output(l, output_layer_shape) for l in y_train])
    # y_test = np.asarray(
    #     [convert_output(l, output_layer_shape) for l in y_test])

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
    epochs = 500  # Production run should have 500
    eta = "inverse"  # Options: float, 'inverse'

    # Default hyperparameters
    default_activation = "sigmoid"
    default_output_activation = "softmax"
    default_cost_function = "log_loss"
    default_learning_rate = "inverse"
    default_eta0 = 0.001
    # TODO: Different regularization runs as well?
    default_regularization = "l2"
    default_mini_batch_size = 20
    default_hidden_layer_size = 10
    default_weight_init = "default"
    default_lambda_value = 0.0
    default_epochs = 500  # Change to 500 for production run
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
    activations = ["sigmoid", "relu", "tanh", "heaviside"]
    output_activations = ["sigmoid", "identity", "softmax"]
    cost_functions = ["mse", "log_loss"]  # , "exponential_cost"]
    learning_rates = np.logspace(-6, -1, 6)
    mini_batch_sizes = [5, 10, 20, 30]
    layer_neurons = [1, 5, 10, 20, 30, 40]
    training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    weight_inits = ["default", "large"]
    lambda_values = np.logspace(-4, 3, 8)

    # # Test run parameters!
    # learning_rates = np.logspace(-6, -1, 6)[:3]
    # mini_batch_sizes = [5, 10, 20]  # , 30, 40, 50]
    # layer_neurons = [5, 10, 15, 20]  # , 25, 30, 40, 50]
    # training_sizes = [0.1, 0.2, 0.3, 0.4]  # , 0.5, 0.6, 0.7, 0.8, 0.9]
    # weight_inits = ["default", "large"]
    # lambda_values = np.logspace(-5, 4, 10)[:5]

    # lmbda_eta_params = default_input_dict.copy()
    # lmbda_eta_params.pop("lmbda")
    # lmbda_eta_params.pop("learning_rate")
    # lmbda_eta_params["figure_folder"] = figure_path
    # run_lambda_eta(X_train, X_test, y_train, y_test, default_layers,
    #                lmbdas=lambda_values, learning_rates=learning_rates,
    #                try_get_pickle=try_get_pickle,
    #                **lmbda_eta_params)

    # Currently running on stationary
    # lmbda_neurons_params = default_input_dict.copy()
    # lmbda_neurons_params.pop("lmbda")
    # lmbda_neurons_params["figure_folder"] = figure_path
    # run_lambda_neurons(X_train, X_test, y_train, y_test, default_layers,
    #                    lmbdas=lambda_values, neurons=layer_neurons,
    #                    try_get_pickle=try_get_pickle,
    #                    **lmbda_neurons_params)

    # TODO: run this!
    # lmbda_mini_batches_params = default_input_dict.copy()
    # lmbda_mini_batches_params.pop("lmbda")
    # lmbda_mini_batches_params.pop("mini_batch_size")
    # lmbda_mini_batches_params["figure_folder"] = figure_path
    # run_lambda_mini_batches(X_train, X_test, y_train, y_test, default_layers,
    #                         lmbdas=lambda_values,
    #                         mini_batch_sizes=mini_batch_sizes,
    #                         try_get_pickle=try_get_pickle,
    #                         **lmbda_mini_batches_params)

    # TODO: run this!
    neurons_eta_params = default_input_dict.copy()
    neurons_eta_params.pop("learning_rate")
    neurons_eta_params["figure_folder"] = figure_path
    run_neurons_eta(X_train, X_test, y_train, y_test, default_layers,
                    neurons=layer_neurons,
                    learning_rates=learning_rates,
                    try_get_pickle=try_get_pickle,
                    **neurons_eta_params)

    # Currently running on mac
    # neurons_training_size_params = default_input_dict.copy()
    # neurons_training_size_params["figure_folder"] = figure_path
    # run_neurons_training_size(default_layers, layer_neurons, training_sizes,
    #                           data_size, data_path,
    #                           try_get_pickle=try_get_pickle,
    #                           **neurons_training_size_params)

    # epoch_weight_init_params = default_input_dict.copy()
    # epoch_weight_init_params.pop("epochs")
    # epoch_weight_init_params.pop("weight_init")
    # run_epoch_weight_init(X_train, X_test, y_train, y_test, default_layers,
    #                       epochs, weight_inits,
    #                       try_get_pickle=try_get_pickle,
    #                       **epoch_weight_init_params)

    # epoch_cost_function_params = default_input_dict.copy()
    # epoch_cost_function_params.pop("epochs")
    # epoch_cost_function_params.pop("cost_function")
    # run_epoch_cost_functions(X_train, X_test, y_train, y_test, default_layers,
    #                          epochs, cost_functions,
    #                          try_get_pickle=try_get_pickle,
    #                          **epoch_cost_function_params)

    # epoch_activations_params = default_input_dict.copy()
    # epoch_activations_params.pop("epochs")
    # epoch_activations_params.pop("activation")
    # epoch_activations_params["cost_function"] = "log_loss"
    # run_epoch_activations(X_train, X_test, y_train, y_test, default_layers,
    #                       epochs, activations,
    #                       try_get_pickle=try_get_pickle,
    #                       **epoch_activations_params)
    
    # epoch_activations_params["cost_function"] = "mse"
    # run_epoch_activations(X_train, X_test, y_train, y_test, default_layers,
    #                       epochs, activations,
    #                       try_get_pickle=try_get_pickle,
    #                       **epoch_activations_params)

    # The following run produces near perfect accuracy
    # nn_core(X_train, X_test, y_train, y_test, default_layers, **default_input_dict)


def run_epoch_activations(X_train, X_test, y_train, y_test, layers,
                          epochs, activations,
                          try_get_pickle=True,
                          figure_folder="../fig", **kwargs):
    """Compares different layer activations."""
    param_dict = {"activations": activations}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    # Double for-loop for all results
    pickle_fname = "mlp_epoch_activations_{}_results.pkl".format(kwargs["cost_function"])

    if os.path.isfile(pickle_fname) and try_get_pickle:
        data = load_pickle(pickle_fname)
    else:
        data = {act: {} for act in activations}
        for i, act in enumerate(activations):
            print("Activation: {}".format(act))
            res_ = nn_core(X_train, X_test, y_train, y_test, layers,
                           activation=act, return_weights=True,
                           epochs=epochs, **kwargs)

            data[act] = convert_nn_core_to_dict(res_)
            data[act]["label"] = act.capitalize()
            data[act]["x"] = np.arange(epochs)
            data[act]["y"] = \
                np.array(data[act]["epoch_evaluations"]) / X_test.shape[0]

        save_pickle(pickle_fname, data)

    figname = "mlp_epoch_activations_{}.pdf".format(kwargs["cost_function"])

    plot_epoch_accuracy(data, r"Epoch", r"Accuracy",
                        figname, vmin=0.0, vmax=1.0)


def run_epoch_cost_functions(X_train, X_test, y_train, y_test, layers,
                             epochs, cost_functions,
                             try_get_pickle=True,
                             figure_folder="../fig", **kwargs):
    """Compares cost functions over epochs inits: mse, log-loss"""
    param_dict = {"cost_functions": cost_functions}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    # Double for-loop for all results
    pickle_fname = "mlp_epoch_cost_functions_results.pkl"

    if os.path.isfile(pickle_fname) and try_get_pickle:
        data = load_pickle(pickle_fname)
    else:
        data = {cf: {} for cf in cost_functions}
        for i, cf in enumerate(cost_functions):
            print("Cost function: {}".format(cf))
            res_ = nn_core(X_train, X_test, y_train, y_test, layers,
                           cost_function=cf, return_weights=True,
                           epochs=epochs, **kwargs)

            data[cf] = convert_nn_core_to_dict(res_)
            data[cf]["label"] = cf.capitalize()
            data[cf]["x"] = np.arange(epochs)
            data[cf]["y"] = \
                np.array(data[cf]["epoch_evaluations"]) / X_test.shape[0]

        save_pickle(pickle_fname, data)

    figname = "mlp_epoch_cost_functions.pdf"

    plot_epoch_accuracy(data, r"Epoch", r"Accuracy",
                        figname, vmin=0.0, vmax=1.0)


def run_epoch_weight_init(X_train, X_test, y_train, y_test, layers,
                          epochs, weight_inits,
                          figure_folder="../fig",
                          try_get_pickle=True,
                          **kwargs):
    """Compares two weight inits."""
    param_dict = {"weight_inits": weight_inits}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    # Double for-loop for all results
    pickle_fname = "mlp_epoch_weight_inits_results.pkl"

    if os.path.isfile(pickle_fname) and try_get_pickle:
        data = load_pickle(pickle_fname)
    else:
        data = {wi: {} for wi in weight_inits}
        for i, wi in enumerate(weight_inits):
            print("Weight init: {}".format(wi))
            res_ = nn_core(X_train, X_test, y_train, y_test, layers,
                           weight_init=wi, return_weights=True,
                           epochs=epochs, **kwargs)

            data[wi] = convert_nn_core_to_dict(res_)
            data[wi]["label"] = wi.capitalize()
            data[wi]["x"] = np.arange(epochs)
            data[wi]["y"] = \
                np.array(data[wi]["epoch_evaluations"]) / X_test.shape[0]

        save_pickle(pickle_fname, data)

    figname = "mlp_epoch_weight_inits.pdf"

    plot_epoch_accuracy(data, r"Epoch", r"Accuracy",
                        figname, vmin=0.0, vmax=1.0)


def run_lambda_mini_batches(X_train, X_test, y_train, y_test, layers,
                            lmbdas=None, mini_batch_sizes=None,
                            try_get_pickle=True,
                            figure_folder="../fig", **kwargs):
    """Compares mini batch sizes for lambda values."""

    param_dict = {"lmbdas": lmbdas, "mini_batch_sizes": mini_batch_sizes}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    # Double for-loop for all results
    pickle_fname = "mlp_lambda_mini_batch_sizes_results.pkl"

    if os.path.isfile(pickle_fname) and try_get_pickle:
        data = load_pickle(pickle_fname)
    else:
        data = {lmbda: {mb: {} for mb in mini_batch_sizes} for lmbda in lmbdas}
        for i, lmbda in enumerate(lmbdas):
            for j, mb in enumerate(mini_batch_sizes):
                print("Lambda: {} MB: {}".format(lmbda, mb))
                res_ = nn_core(X_train, X_test, y_train, y_test, layers,
                               lmbda=lmbda,
                               mini_batch_size=mb,
                               return_weights=True,
                               **kwargs)
                data[lmbda][mb] = res_

        save_pickle(pickle_fname, data)

    # Maps values to matrix
    plot_data = np.empty((len(lmbdas), len(mini_batch_sizes)))

    # Populates plot data
    for i, lmbda in enumerate(lmbdas):
        for j, mb in enumerate(mini_batch_sizes):
            plot_data[i, j] = data[lmbda][mb][1]

    heatmap_plotter(lmbdas, mini_batch_sizes, plot_data.T,
                    "mlp_lambda_mini_batch_size.pdf",
                    tick_param_fs=8, label_fs=10,
                    vmin=0.0, vmax=1.0, xlabel=r"$\lambda$",
                    ylabel=r"$N_\mathrm{MB}$",
                    cbartitle=r"Accuracy",
                    x_tick_mode="exp", y_tick_mode="int")


def run_lambda_neurons(X_train, X_test, y_train, y_test, layers,
                       lmbdas=None, neurons=None,
                       try_get_pickle=True,
                       figure_folder="../fig", **kwargs):
    """Compares different lambdas for different neuron sizes."""

    param_dict = {"lmbdas": lmbdas, "neurons": neurons}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    # Double for-loop for all results
    pickle_fname = "mlp_lambda_neurons_results.pkl"

    if os.path.isfile(pickle_fname) and try_get_pickle:
        data = load_pickle(pickle_fname)
    else:
        data = {lmbda: {neuron: {} for neuron in neurons} for lmbda in lmbdas}
        for i, lmbda in enumerate(lmbdas):
            for j, neuron in enumerate(neurons):
                print("Lambda: {} Neuron: {}".format(lmbda, neuron))
                layers[1] = neuron
                res_ = nn_core(X_train, X_test, y_train, y_test, layers,
                               lmbda=lmbda, return_weights=True,
                               **kwargs)
                data[lmbda][neuron] = res_

        save_pickle(pickle_fname, data)

    # Maps values to matrix
    plot_data = np.empty((len(lmbdas), len(neurons)))

    # Populates plot data
    for i, lmbda in enumerate(lmbdas):
        for j, n in enumerate(neurons):
            plot_data[i, j] = data[lmbda][n][1]

    heatmap_plotter(lmbdas, neurons, plot_data.T, "mlp_lambda_neurons.pdf",
                    tick_param_fs=8, label_fs=10,
                    xlabel=r"$\lambda$",
                    ylabel=r"Neurons", vmin=0.0, vmax=1.0,
                    cbartitle=r"Accuracy", x_tick_mode="exp",
                    y_tick_mode="int")


def run_neurons_eta(X_train, X_test, y_train, y_test, layers,
                    neurons=None, learning_rates=None,
                    try_get_pickle=True,
                    figure_folder="../fig", **kwargs):
    """Compares different neuron sizes for different etas."""

    param_dict = {"neurons": neurons, "learning_rates": learning_rates}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    # Double for-loop for all results
    pickle_fname = "mlp_neurons_eta_results.pkl"

    if os.path.isfile(pickle_fname) and try_get_pickle:
        data = load_pickle(pickle_fname)
    else:
        data = {n: {eta: {} for eta in learning_rates} for n in neurons}
        for i, neuron in enumerate(neurons):
            for j, eta in enumerate(learning_rates):
                print("Neuron: {} Eta: {}".format(neuron, eta))
                layers[1] = neuron
                res_ = nn_core(X_train, X_test, y_train, y_test, layers,
                               return_weights=True,
                               learning_rate=eta, **kwargs)
                data[neuron][eta] = res_

        save_pickle(pickle_fname, data)

    # Maps values to matrix
    plot_data = np.empty((len(neurons), len(learning_rates)))

    # Populates plot data
    for i, n in enumerate(neurons):
        for j, eta in enumerate(learning_rates):
            plot_data[i, j] = data[n][eta][1]

    heatmap_plotter(neurons, learning_rates, plot_data.T,
                    "mlp_neurons_eta.pdf",
                    tick_param_fs=8, label_fs=10,
                    xlabel=r"Neurons", ylabel=r"$\eta$",
                    cbartitle=r"Accuracy", vmin=0.0, vmax=1.0,
                    x_tick_mode="int", y_tick_mode="exp")


def run_neurons_training_size(layers, neurons, training_sizes,
                              data_size, data_path, try_get_pickle=True,
                              figure_folder="../fig", **kwargs):
    """Compares different neurons for different training sizese."""
    param_dict = {"neurons": neurons, "training_sizes": training_sizes}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    # Double for-loop for all results
    pickle_fname = "mlp_neurons_training_size_results.pkl"

    if os.path.isfile(pickle_fname) and try_get_pickle:
        data = load_pickle(pickle_fname)
    else:
        data = {n: {ts: {} for ts in training_sizes} for n in neurons}
        for i, neuron in enumerate(neurons):
            for j, ts in enumerate(training_sizes):
                inlay, outlay, X_train, X_test, y_train, y_test = \
                    retrieve_2d_data_formatted(data_path, data_size, ts)

                print("Neurons: {} Training size: {}".format(neuron, ts))
                layers[1] = neuron
                print(X_train.shape, X_test.shape)
                res_ = nn_core(X_train, X_test, y_train, y_test, layers,
                               return_weights=True, **kwargs)
                data[neuron][ts] = res_

        save_pickle(pickle_fname, data)

    # Maps values to matrix
    plot_data = np.empty((len(neurons), len(training_sizes)))

    # Populates plot data
    for i, n in enumerate(neurons):
        for j, ts in enumerate(training_sizes):
            plot_data[i, j] = data[n][ts][1]

    heatmap_plotter(neurons, training_sizes, plot_data.T,
                    "mlp_neurons_training_size.pdf",
                    tick_param_fs=8, label_fs=10,
                    xlabel=r"Neurons", ylabel=r"Training size",
                    cbartitle=r"Accuracy",  vmin=0.0, vmax=1.0,
                    x_tick_mode="int", y_tick_mode="float")


def run_lambda_eta(X_train, X_test, y_train, y_test, layers,
                   lmbdas=None, learning_rates=None, try_get_pickle=True,
                   figure_folder="../fig", **kwargs):
    """Runs NN for different lambdas and etas."""
    param_dict = {"lmbdas": lmbdas, "learning_rates": learning_rates}
    param_dict.update(kwargs)
    print_parameters(**param_dict)

    # Double for-loop for all results
    pickle_fname = "mlp_lambda_eta_results.pkl"

    if os.path.isfile(pickle_fname) and try_get_pickle:
        data = load_pickle(pickle_fname)
    else:
        data = {lmbda: {eta: {} for eta in learning_rates} for lmbda in lmbdas}
        for i, lmbda in enumerate(lmbdas):
            for j, eta in enumerate(learning_rates):
                print("Lambda: {} Eta: {}".format(lmbda, eta))
                res_ = nn_core(X_train, X_test, y_train, y_test, layers,
                               lmbda=lmbda, return_weights=True,
                               learning_rate=eta, **kwargs)
                data[lmbda][eta] = res_

        save_pickle(pickle_fname, data)

    # Maps values to matrix
    plot_data = np.empty((len(lmbdas), len(learning_rates)))

    # Populates plot data
    for i, lmbda in enumerate(lmbdas):
        for j, eta in enumerate(learning_rates):
            plot_data[i, j] = data[lmbda][eta][1]

    heatmap_plotter(lmbdas, learning_rates, plot_data.T, "mlp_lambda_eta.pdf",
                    tick_param_fs=8, label_fs=10,
                    xlabel=r"$\lambda$", ylabel=r"$\eta$",
                    cbartitle=r"Accuracy", vmin=0.0, vmax=1.0,
                    x_tick_mode="exp", y_tick_mode="exp")


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


def retrieve_2d_data_formatted(data_path, data_size, training_size):
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

    return input_layer_shape, output_layer_shape, X_train, X_test, y_train, \
        y_test


def run_optimum_parameters(X_train, X_test, y_train, y_test,
                           default_layers, **default_input_dict):
    res_ = nn_core(X_train, X_test, y_train, y_test,
                   default_layers, **default_input_dict)


if __name__ == '__main__':
    task1e()
