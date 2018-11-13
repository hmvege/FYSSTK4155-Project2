#!/usr/bin/env python3

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from lib import ising_1d as ising
from lib import neuralnetwork as nn


def read_t(t="all", root="."):
    """Loads an ising model data set."""
    if t == "all":
        data = pickle.load(open(os.path.join(
            root, "Ising2DFM_reSample_L40_T=All.pkl"), "rb"))
    else:
        data = pickle.load(open(os.path.join(
            root, "Ising2DFM_reSample_L40_T=%.2f.pkl".format(t)), "rb"))

    return np.unpackbits(data).astype(int).reshape(-1, 1600)


def load_pickle(pickle_file_name):
    """Loads a pickle from given pickle_file_name."""
    with open(pickle_file_name, "rb") as f:
        data = pickle.load(f)
        print("Pickle file loaded: {}".format(pickle_file_name))
    return data


def save_pickle(pickle_file_name, data):
    """Saves data as a pickle."""
    with open(pickle_file_name, "wb") as f:
        pickle.dump(data, f)
        print("Data pickled and dumped to: {:s}".format(pickle_file_name))


def print_parameters(**kwargs):
    """Prints run parameters."""
    print("\nRUN PARAMETERS:")
    for key, val in kwargs.items():
        print("{0:20s}: {1:20s}".format(key.capitalize(), str(val)))


def plot_accuracy_scores(lmbdas, train_accuracy_values, test_accuracy_values,
                         labels, figname, xlabel, ylabel,
                         figure_folder="../fig"):
    """General accuracy plotter."""
    linestyles = [
        ('solid',               (0, ())),
        ('loosely dotted',      (0, (1, 10))),
        ('dotted',              (0, (1, 5))),
        ('densely dotted',      (0, (1, 1))),

        ('loosely dashed',      (0, (5, 10))),
        ('dashed',              (0, (5, 5))),
        ('densely dashed',      (0, (5, 1))),

        ('loosely dashdotted',  (0, (3, 10, 1, 10))),
        ('dashdotted',          (0, (3, 5, 1, 5))),
        ('densely dashdotted',  (0, (3, 1, 1, 1))),

        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
              "#e6ab02", "#a6761d", "#666666"]

    markers = [".", "o", "v", "^", "1", "s", "*", "x", "+", "D", "p", ">", "<"]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    for train_val, test_val, lab, ls1, ls2, col, mk1, mk2 in zip(
            train_accuracy_values, test_accuracy_values, labels,
            linestyles[len(linestyles)//2:], linestyles[:len(linestyles)//2],
            colors, markers[len(markers)//2:], markers[:len(markers)//2]):
        ax1.semilogx(lmbdas, train_val,
                     marker=mk1, ls=ls1[-1],
                     color=col,
                     label=lab + r" train")
        ax1.semilogx(lmbdas, test_val,
                     marker=mk2, ls=ls2[-1],
                     color=col,
                     label=lab + r" test")

    ax1.set_xlabel(ylabel)
    ax1.set_ylabel(xlabel)

    ax1.grid(True)
    ax1.legend(fontsize=8)

    figure_path = os.path.join(figure_folder, "{}.pdf".format(figname))
    fig.savefig(figure_path)
    print("Figure saved at {}".format(figure_path))
    plt.close(fig)


def convert_output(label_, output_size):
    """Converts label to output vector."""
    y_ = np.zeros(output_size, dtype=float)
    y_[label_] = 1.0
    return y_


def retrieve_2d_ising_data(data_path, data_size):
    """Simple retriever."""

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
    Y_ordered = labels_data[:70000]
    X_critical = input_data[70000:100000, :]
    Y_critical = labels_data[70000:100000]
    X_disordered = input_data[100000:, :]
    Y_disordered = labels_data[100000:]

    # X_ordered = input_data[:int(np.floor(70000*data_percentage)), :]
    # Y_ordered = labels_data[:int(np.floor(70000*data_percentage))]

    # X_critical = input_data[70000:int(np.floor(100000*data_percentage)), :]
    # Y_critical = labels_data[70000:int(np.floor(100000*data_percentage))]

    # X_disordered = input_data[100000:int(
    #     np.floor(100000*(1 + data_percentage))), :]
    # Y_disordered = labels_data[100000:int(
    #     np.floor(100000*(1 + data_percentage)))]

    del input_data, labels_data

    # define training and test data sets
    X = np.concatenate((X_ordered[:data_size//2], X_disordered[:data_size//2]))
    Y = np.concatenate((Y_ordered[:data_size//2], Y_disordered[:data_size//2]))

    # Splits 50/50 into ordered, disordered states

    return X, Y


def nn_core(X_train, X_test, y_train, y_test,
            layers, lmbda=None,
            activation=None,
            output_activation=None,
            cost_function=None,
            learning_rate=None,
            eta0=None,
            regularization=None,
            weight_init=None,
            epochs=None,
            mini_batch_size=None,
            tolerance=None, 
            return_weights=False, 
            verbose=False):
    """Method for retrieveing data for a given set of hyperparameters

    Args:
        X_train (ndarray)
        X_test (ndarray)
        y_train (ndarray)
        y_test (ndarray)
        layers (list(int)): list of layer sizes
        lmbdas (float): list of lmbdas
        activation (str): activation function.
        output_activation (str): output activation function. Choices: 
            sigmoidal, softmax, identity.
        learning_rate (str|float): learning rate. Options: float, inverse
        eta0 (float): start eta for 'inverse'.
        regularization (str): regularization, choices: l1, l2, elastic_net
        mini_batch_size (float): minibatch size
        tolerance (float): tolerance, at what point we cut off the parameter 
            search.
        return_weights (bool): returns the weight matrices
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
                                  regularization=regularization,
                                  alpha=lmbda)


    MLP.train(X_train, y_train,
              data_test=X_test,
              data_test_labels=y_test,
              mini_batch_size=mini_batch_size,
              epochs=epochs,
              eta=learning_rate,
              eta0=eta0,
              verbose=verbose)

    # Results list
    res = []

    if (not isinstance(X_test, type(None))) and \
        (not isinstance(y_test, type(None))):

        # Accuracy score for our implementation
        train_accuracy = MLP.score(X_train, y_train)
        test_accuracy = MLP.score(X_test, y_test)

        train_accuracy_epochs = MLP.evaluate(X_train, y_train)
        test_accuracy_epochs = MLP.evaluate(X_test, y_test)
        # critical_accuracy[i]=log_reg.score(X_critical,Y_critical)

        # Prints result from single lambda run
        if verbose:
            print('Accuracy scores: train, test')
            print('MultilayerPerceptron: {0:.3f}, {1:.3f}'.format(
                train_accuracy, test_accuracy))

        res += [train_accuracy, test_accuracy, train_accuracy_epochs,
               test_accuracy_epochs]

    if return_weights:
        res.append(MLP.weights)
        res.append(MLP.biases)
        res.append(MLP.epoch_evaluations)

    return res


def plot_heatmap(J_leastsq, J_ridge, J_lasso, L, lmbda, figure_folder,
                 filename):
    """Plots and saves a heatmap for task b) and d)."""
    cmap_args = dict(vmin=-1., vmax=1., cmap='seismic')

    fig, axarr = plt.subplots(nrows=1, ncols=3)

    fontsize = 8
    labelsize = 10
    yticksize = 8

    axarr[0].imshow(J_leastsq, **cmap_args)
    axarr[0].set_title(r'$\mathrm{OLS}$', fontsize=fontsize)
    axarr[0].tick_params(labelsize=labelsize)

    axarr[1].imshow(J_ridge, **cmap_args)
    axarr[1].set_title(
        r'$\mathrm{Ridge}, \lambda=%.4f$' % (lmbda), fontsize=fontsize)
    axarr[1].tick_params(labelsize=labelsize)

    im = axarr[2].imshow(J_lasso, **cmap_args)
    axarr[2].set_title(
        r'$\mathrm{LASSO}, \lambda=%.4f$' % (lmbda), fontsize=fontsize)
    axarr[2].tick_params(labelsize=labelsize)

    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)

    cbar.ax.set_yticklabels(
        np.arange(-1.0, 1.0+0.25, 0.25), fontsize=yticksize)
    cbar.set_label(r'$J_{i,j}$', labelpad=-40,
                   y=1.12, fontsize=fontsize, rotation=0)

    # plt.show()
    figure_path = os.path.join(
        figure_folder, filename)
    fig.savefig(figure_path)
    print("Figure for lambda={} stored at {}.".format(lmbda, figure_path))

    plt.close(fig)


def plot_all_r2(lmbda_values, r2_ols_test, r2_ols_train, r2_ridge_test,
                r2_ridge_train, r2_lasso_test, r2_lasso_train, figname,
                figure_folder, verbose=False):
    """Plots all r2 scores together."""
    if verbose:
        print("OLS R2 test:", r2_ols_test)
        print("OLS R2 train:", r2_ols_train)
        print("Ridge R2 test:", r2_ridge_test)
        print("Ridge R2 train:", r2_ridge_train)
        print("Lasso R2 test:", r2_lasso_test)
        print("Lasso R2 train:", r2_lasso_train)

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    # OLS
    ax1.axhline(r2_ols_test, label=r"OLS test",
                marker="", ls=(0, (3, 1, 1, 1)),  # Densely dashdotted
                color="#7570b3")
    ax1.axhline(r2_ols_train, label=r"OLS train",
                marker="", ls=(0, (3, 1, 1, 1)),  # Densely dashdotted
                color="#7570b3")

    # Ridge
    ax1.semilogx(lmbda_values, r2_ridge_test, label=r"Ridge test",
                 marker="o", ls=(0, (5, 1)),  # Densely dashed
                 color="#1b9e77")
    ax1.semilogx(lmbda_values, r2_ridge_train, label=r"Ridge train",
                 marker="x", ls=(0, (5, 1)),  # Densely dashed
                 color="#1b9e77")

    # Lasso
    ax1.semilogx(lmbda_values, r2_lasso_test, label=r"Lasso test",
                 marker="o", ls=(0, (3, 5, 1, 5)),  # Dashdotted
                 color="#d95f02")
    ax1.semilogx(lmbda_values, r2_lasso_train, label=r"Lasso train",
                 marker="x", ls=(0, (3, 5, 1, 5)),  # Dashdotted
                 color="#d95f02")

    ax1.set_xlim(lmbda_values[0], lmbda_values[-1])
    ax1.set_ylim(-0.05,1.05)
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel(r"$R^2$")
    ax1.legend()
    ax1.grid(True)

    figure_path = os.path.join(figure_folder, "{}.pdf".format(figname))
    fig.savefig(figure_path)
    print("Figure saved at {}".format(figure_path))


def heatmap_plotter(x, y, z, figure_name, tick_param_fs=None, label_fs=None,
                    vmin=None, vmax=None, xlabel=None, ylabel=None,
                    cbartitle=None):
    """Plots a heatmap surface."""
    fig, ax = plt.subplots()

    yheaders = ['%1.2e' % i for i in y]
    xheaders = ['%1.1e' % i for i in x]

    # X, Y = np.meshgrid(x,y)
    # print (X.shape, Y.shape, z.shape)

    heatmap = ax.pcolor(z, edgecolors="k", linewidth=2, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.ax.tick_params(labelsize=tick_param_fs)
    cbar.ax.set_title(cbartitle, fontsize=label_fs)

    # # ax.set_title(method, fontsize=fs1)
    ax.set_xticks(np.arange(z.shape[1]) + .5, minor=False)
    ax.set_yticks(np.arange(z.shape[0]) + .5, minor=False)

    ax.set_xticklabels(xheaders, rotation=90, fontsize=tick_param_fs)
    ax.set_yticklabels(yheaders, fontsize=tick_param_fs)

    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    plt.tight_layout()

    fig.savefig(os.path.join("../fig", figure_name))
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)
