import numpy as np
import copy as cp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn

from lib import ising_1d as ising
from lib import regression as reg
from lib import metrics
from lib import bootstrap as bs
from lib import cross_validation as cv

import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils


def task1b():

    # Number of samples to generate
    N_samples = 10000
    training_size = 0.1

    np.random.seed(12)

    # system size
    L = 40

    # create 10000 random Ising states
    states = np.random.choice([-1, 1], size=(N_samples, L))

    # calculate Ising energies
    energies = ising.ising_energies(states, L)

    # reshape Ising states into RL samples: S_iS_j --> X_p
    states=np.einsum('...i,...j->...ij', states, states)

    # Reshaping to correspond to energies.
    # Shamelessly stolen a lot of from:
    # https://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB_CVI-linreg_ising.html
    # E.g. why did no-one ever tell me about einsum? 
    # That's awesome - no way I would have discovered that by myself.
    shape=states.shape
    states=states.reshape((shape[0],shape[1]*shape[2]))

    # # build final data set
    # Data=[states,energies]

    # # # define number of samples
    # n_samples=400

    # # define train and test data sets
    # X_train=Data[0][:n_samples]
    # y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
    # X_test=Data[0][n_samples:3*n_samples//2]
    # y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])

    # # define error lists
    # train_errors_leastsq = []
    # test_errors_leastsq = []

    # train_errors_ridge = []
    # test_errors_ridge = []

    # train_errors_lasso = []
    # test_errors_lasso = []

    # #Initialize coeffficients for ridge regression and Lasso
    # coefs_leastsq = []
    # coefs_ridge = []
    # coefs_lasso=[]

    X_train, X_test, y_train, y_test = \
        sk_modsel.train_test_split(states, energies, test_size=1-training_size)

    lambda_values = np.logspace(-4,5,10)

    # Linear regression
    linreg = reg.OLSRegression()
    linreg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
    y_pred_linreg = linreg.predict(cp.deepcopy(X_test))

    print("LINREG:")
    print("R2:  {:-20.16f}".format(metrics.r2(y_test, y_pred_linreg)))
    print("MSE: {:-20.16f}".format(metrics.mse(y_test, y_pred_linreg)))
    print("Bias: {:-20.16f}".format(metrics.bias(y_test, y_pred_linreg)))
    print("Beta coefs: {}".format(linreg.coef_))
    print("Beta coefs variances: {}".format(linreg.coef_var))

    J_leastsq = np.asarray(linreg.coef_).reshape((L,L))

    BootstrapWrapper(x, y, design_matrix, reg, N_bs, test_percent=0.4)

    for lmbda in lambda_values:

        # Ridge regression
        ridge_reg = reg.RidgeRegression(lmbda)
        ridge_reg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
        y_pred_ridge = ridge_reg.predict(cp.deepcopy(X_test))

        print("\nRIDGE:")
        print("R2:  {:-20.16f}".format(metrics.r2(y_test, y_pred_ridge)))
        print("MSE: {:-20.16f}".format(metrics.mse(y_test, y_pred_ridge)))
        print("Bias: {:-20.16f}".format(metrics.bias(y_test, y_pred_ridge)))
        print("Beta coefs: {}".format(ridge_reg.coef_))
        print("Beta coefs variances: {}".format(ridge_reg.coef_var))


        # Lasso regression
        lasso_reg = sk_model.Lasso(alpha=lmbda)
        lasso_reg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
        y_pred_lasso = lasso_reg.predict(cp.deepcopy(X_test))

        print("\nLASSO:")
        print("R2:  {:-20.16f}".format(metrics.r2(y_test, y_pred_lasso)))
        print("MSE: {:-20.16f}".format(metrics.mse(y_test, y_pred_lasso)))
        print("Bias: {:-20.16f}".format(metrics.bias(y_test, y_pred_lasso)))
        print("Beta coefs: {}".format(lasso_reg.coef_))

        J_ridge = np.asarray(ridge_reg.coef_).reshape((L,L))
        J_lasso = np.asarray(lasso_reg.coef_).reshape((L,L))


        cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

        fig, axarr = plt.subplots(nrows=1, ncols=3)
        
        axarr[0].imshow(J_leastsq,**cmap_args)
        # axarr[0].set_title(r'$\mathrm{OLS}$',fontsize=16)
        # axarr[0].tick_params(labelsize=16)

        axarr[1].imshow(J_ridge,**cmap_args)
        # axarr[1].set_title(r'$\mathrm{Ridge}, \lambda=%.4f$' %(lmbda),fontsize=16)
        # axarr[1].tick_params(labelsize=16)

        im=axarr[2].imshow(J_lasso,**cmap_args)
        # axarr[2].set_title(r'$\mathrm{LASSO}, \lambda=%.4f$' %(lmbda),fontsize=16)
        # axarr[2].tick_params(labelsize=16)

        # divider = make_axes_locatable(axarr[2])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar=fig.colorbar(im, cax=cax)

        # cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
        # cbar.set_label(r'$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)

        # fig.subplots_adjust(right=2.0)

        plt.show()

        plt.close(fig)


def main():
    task1b()



if __name__ == '__main__':
    main()