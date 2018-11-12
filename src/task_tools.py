#!/usr/bin/env python3

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from lib import ising_1d as ising


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


def retrieve_2d_ising_data(data_path, data_percentage):
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

    print(int(np.floor(70000*data_percentage)), int(np.floor(100000*(1 + data_percentage))))

    return X, Y



class StoreValues:
    def __init__(self, name):
        self.name = name
        self.values_dict = {}

    def update(self, value_dict):
        self.values_dict.update(value_dict)
