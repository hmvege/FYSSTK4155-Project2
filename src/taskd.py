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




def run_nn_fit(X_train, X_test, y_train, y_test,
                   lmbdas=[None], penalties=[None],
                   activations=[None], solvers=[None],
                   learning_rates=[None],
                   momentums=[None], mini_batch_sizes=[None],
                   max_iter=None, tolerances=[None], verbose=False):
    """Method for retrieveing data for given lists of hyperparameters

    Args:
        X_train (ndarray)
        X_test (ndarray)
        y_train (ndarray)
        y_test (ndarray)
        lmbdas (list(float)): list of lmbdas.
        penalties (list(str): list of penalty types. Choices: l1, l2, 
            elastic_net.
        activations (list(str)): list of activation functions.
        solvers (list(str)): list of solver functions.
        learning_rates (list(str|float)): list of learning rates. 
            Options: float, inverse.
        momentums (list(float)): momentum strengths, list of floats.
        mini_batch_sizes (list(float)): list of minibatch sizes.
        max_iter (int): maximum number of iterations
        tolerances (list(floats)): list of tolerances
        verbose (bool): more verbose output. Default is False.

    Returns:
        Dictionary with logreg accuracy scores and times
    """

    # Create dict to populate
    for penalty in penalties:
        for activation in activations:
            for solver in solvers:
                for lr in learning_rates:
                    for momentum in momentums:
                        for mb in mini_batch_sizes:
                            for tol in tolerances:    

    for penalty in penalties:
        for activation in activations:
            for solver in solvers:
                for lr in learning_rates:
                    for momentum in momentums:
                        for mb in mini_batch_sizes:
                            for tol in tolerances:
                                logreg_core(X_train, X_test,
                                            y_train, y_test,
                                            lmbdas=lmbdas,
                                            penalty=penalty,
                                            activation=activation,
                                            learning_rate=lr,
                                            momentum=momentum,
                                            mini_batch_size=mb,
                                            max_iter=max_iter,
                                            tolerance=tol,
                                            verbose=False)


def nn_core(X_train, X_test, y_train, y_test,
                layers, lmbdas=[None], penalty=None,
                activation=None, solver=None, learning_rate=None,
                momentum=None, mini_batch_size=None, max_iter=None,
                tolerance=None, verbose=False):
    """Method for retrieveing data for given lists of hyperparameters

    Args:
        X_train (ndarray)
        X_test (ndarray)
        y_train (ndarray)
        y_test (ndarray)
        layers (list(int)): list of layer sizes
        lmbdas (float): list of lmbdas
        penalty (str): penalty type. Choices: l1, l2, elastic_net
        activation (str): activation function.
        solver(str): solver function.
        learning_rate (str|float): learning rate. Options: float, inverse
        momentum (float): momentum strength
        mini_batch_size (float): minibatch size
        tolerance (float): tolerance, at what point we cut off the parameter 
            search.
        verbose (bool): more verbose output. Default is False

    Returns:
        Dictionary with logreg accuracy scores and times
        SK-learn dictionary with accuracy scores and times
        SGD-SK-learn dictionary with accuracy scores and times
    """

    # Sets up data arrays
    train_accuracy = np.zeros(lmbdas.shape, np.float64)
    test_accuracy = np.zeros(lmbdas.shape, np.float64)
    critical_accuracy = np.zeros(lmbdas.shape, np.float64)

    # Loops over regularisation strength
    for i, lmbda in enumerate(lmbdas):

        if verbose:
            print("")
            print("="*80)
            print("Lambda = ", lmbda)

        # Our implementation of logistic regression
        # Sets up my MLP.
        MLP = MultilayerPerceptron(layers,
                                   activation=activation,
                                   cost_function=cost_function,
                                   output_activation=output_activation,
                                   weight_init=weight_init,
                                   alpha=alpha)
        MLP.train(X_train, y_train,
                  data_test=X_test,
                  data_test_labels=y_test,
                  mini_batch_size=mini_batch_size,
                  epochs=epochs,
                  eta=eta)

        # Accuracy score for our implementation
        train_accuracy[i] = MLP.score(X_train, y_train)
        test_accuracy[i] = MLP.score(X_test, y_test)
        # critical_accuracy[i]=log_reg.score(X_critical,Y_critical)

        # Prints result from single lambda run
        if verbose:
            print('Accuracy scores: train, test, critical')
            print('MultilayerPerceptron: {0:0.4f}, {1:0.4f}, {2:0.4f}'.format(
                train_accuracy[i], test_accuracy[i], critical_accuracy[i]))

        # Prints iteration values
        if verbose:
            print("Finished computing {}/11 iterations".format(i+1))

    return train_accuracy, test_accuracy, critical_accuracy
