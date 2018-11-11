#!/usr/bin/env python3

import os
import pickle

def task1e(figure_path="../fig"):
    """Task e) of project 2.

    Task: train the NN with the cross entropy function and compare with 
    Logistic Regression results from c).
    """
    print("="*80)
    print("Task e: neural network classification")
    training_size = 0.8
    fract = 0.01
    learning_rate = 1.0
    max_iter = int(1e3)
    tolerance = 1e-8

    print("Logistic regression")

    data_path = "../datafiles/MehtaIsingData"
    input_data = read_t("all", data_path)

    labels_data = pickle.load(open(os.path.join(
        data_path, "Ising2DFM_reSample_L40_T=All_labels.pkl"), "rb"))

    # Use just 10000 of the training data, nothing more

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
    eta = "inverse" # Options: float, 'inverse'

