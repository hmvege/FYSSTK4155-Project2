#!/usr/bin/env python3

import os
import pickle

def task1d(figure_path="../fig"):
    """Task d) of project 2.

    Task: train the NN and compare with Linear Regression results from b).
    """
    print("="*80)
    print("Task d: neural network regression")
    training_size = 0.8
    fract = 0.01
    learning_rate = 1.0
    max_iter = int(1e3)
    tolerance = 1e-8

    data_path = "../datafiles/MehtaIsingData"
    input_data = read_t("all", data_path)

    labels_data = pickle.load(open(os.path.join(
        data_path, "Ising2DFM_reSample_L40_T=All_labels.pkl"), "rb"))

    print("Come back later")
    sys.exit()
