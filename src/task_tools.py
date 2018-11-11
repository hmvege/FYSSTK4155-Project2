#!/usr/bin/env python3

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


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
                         labels, figname, xlabel, ylabel, figure_folder="../fig"):
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
