import numpy as np
import copy as cp

from lib import ising_1d as ising
from lib import regression as reg

import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils


def task1b():
    # Polynomial degree
    deg = 3

    # Number of samples to generate
    N_samples = 10

    np.random.seed(12)

    # system size
    L = 10

    # create 10000 random Ising states
    states = np.random.choice([-1, 1], size=(N_samples, L))

    # calculate Ising energies
    energies = ising.ising_energies(states, L)

    print(states.shape, energies.shape)

    states=np.einsum('...i,...j->...ij', states, states)
    print(states.shape, energies.shape)

    print (states[0])

    exit("Exits @ 37")


    # Linear regression
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(cp.deepcopy(states), energies.ravel())
    
    print (X.shape)

    linreg = reg.OLSRegression()
    linreg.fit(X, cp.deepcopy(energies.ravel()))
    y_pred = linreg.predict(X).ravel()

    print("R2:  {:-20.16f}".format(metrics.R2(energies.ravel(), y_pred)))
    print("MSE: {:-20.16f}".format(metrics.mse(energies.ravel(), y_pred)))
    print("Bias: {:-20.16f}".format(metrics.bias2(energies.ravel(), y_pred)))
    print("Beta coefs: {}".format(linreg.coef_))
    print("Beta coefs variances: {}".format(linreg.coef_var))


    # Lasso regression


    # Ridge regression



def main():
    task1b()



if __name__ == '__main__':
    main()