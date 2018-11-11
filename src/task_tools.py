#!/usr/bin/env python3

import os
import pickle
import numpy as np

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
    print("RUN PARAMETERS:")
    for key, val in kwargs:
        print("{}: {}".format(key.capitalize(), val))

