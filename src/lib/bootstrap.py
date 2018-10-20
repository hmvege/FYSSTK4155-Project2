#!/usr/bin/env python3
import numpy as np
try:
    import lib.metrics as metrics
except ModuleNotFoundError:
    import metrics
from tqdm import tqdm
import sklearn.model_selection as sk_modsel
# import sklearn.cross_validation as sk_cv
import sklearn.utils as sk_utils
import sklearn.metrics as sk_metrics


def boot(*data):
    """Strip-down version of the bootstrap method.

    Args:
        *data (ndarray): list of data arrays to resample.

    Return:
        *bs_data (ndarray): list of bootstrapped data arrays."""

    N_data = len(data)
    N = data[0].shape[0]
    assert np.all(np.array([len(d) for d in data]) == N), \
        "unequal lengths of data passed."

    index_lists = np.random.randint(N, size=N)

    return [d[index_lists] for d in data]


class BootstrapRegression:
    """Bootstrap class intended for use together with regression."""
    _reg = None
    _design_matrix = None

    def __init__(self, x_data, y_data, reg, design_matrix_func):
        """
        Initialises an bootstrap regression object.
        Args:
        """
        assert len(x_data) == len(y_data), "x and y data not of equal lengths"
        self.x_data = x_data
        self.y_data = y_data
        self._reg = reg
        self._design_matrix = design_matrix_func

    @property
    def design_matrix(self):
        return self._design_matrix

    @design_matrix.setter
    def design_matrix(self, f):
        self._design_matrix = f

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, reg):
        self._reg = reg

    @property
    def coef_(self):
        return self.coef_coefs

    @coef_.getter
    def coef_(self):
        return self.beta_coefs

    @property
    def coef_var(self):
        return self.beta_coefs_var

    @coef_var.getter
    def coef_var(self):
        return self.beta_coefs_var

    @metrics.timing_function
    def bootstrap(self, N_bs, test_percent=0.25):
        """
        Performs a bootstrap for a given regression type, design matrix 
        function and excact function.

        Args:
            N_bs (int): number of bootstraps to perform
            test_percent (float): what percentage of data to reserve for 
                testing.
        """

        assert not isinstance(self._reg, type(None))
        assert not isinstance(self._design_matrix, type(None))

        assert test_percent < 1.0, "test_percent must be less than one."

        N = len(self.x_data)

        # Splits into test and train set.
        # test_size = int(np.floor(N * test_percent))

        x = self.x_data
        y = self.y_data

        # # Splits into training and test set.
        # x_test, x_train = np.split(x, [test_size], axis=0)
        # y_test, y_train = np.split(y, [test_size], axis=0)

        # Splits X data and design matrix data
        x_train, x_test, y_train, y_test = \
            sk_modsel.train_test_split(self.x_data, self.y_data,
                                       test_size=test_percent, shuffle=False)
        test_size = x_test.shape[0]

        # Sets up emtpy lists for gathering the relevant scores in
        r2_list = np.empty(N_bs)
        # mse_list = np.empty(N_bs)
        # bias_list = np.empty(N_bs)
        # var_list = np.empty(N_bs)
        beta_coefs = []

        # Sets up design matrix to test for
        X_test = self._design_matrix(x_test)

        y_pred_list = np.empty((test_size, N_bs))

        # Sets up the X_tra
        X_train = self._design_matrix(x_train)

        # Bootstraps
        for i_bs in tqdm(range(N_bs), desc="Bootstrapping"):
            # Bootstraps test data
            # x_boot, y_boot = boot(x_train, y_train)

            X_boot, y_boot = boot(X_train, y_train)
            # Sets up design matrix
            # X_boot = self._design_matrix(x_boot)

            # Fits the bootstrapped values
            self.reg.fit(X_boot, y_boot)

            # Tries to predict the y_test values the bootstrapped model
            y_predict = self.reg.predict(X_test)

            # Calculates r2
            r2_list[i_bs] = metrics.r2(y_test, y_predict)
            # mse_list[i_bs] = metrics.mse(y_predict, y_test)
            # bias_list[i_bs] = metrics.bias(y_predict, y_test)
            # var_list[i_bs] = np.var(y_predict)

            # Stores the prediction and beta coefs.
            y_pred_list[:, i_bs] = y_predict.ravel()
            beta_coefs.append(self.reg.coef_)

        # pred_list_bs = np.mean(y_pred_list, axis=0)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        self.r2 = np.mean(r2_list)

        # Mean Square Error, mean((y - y_approx)**2)
        # _mse = np.mean((y_test.ravel() - y_pred_list)**2,
        #                axis=0, keepdims=True)
        _mse = np.mean((y_test - y_pred_list)**2,
                   axis=1, keepdims=True)
        self.mse = np.mean(_mse)

        # Bias, (y - mean(y_approx))^2
        _y_pred_mean = np.mean(y_pred_list, axis=1, keepdims=True)
        self.bias = np.mean((y_test - _y_pred_mean)**2)

        # Variance, var(y_approx)
        self.var = np.mean(np.var(y_pred_list,
                                  axis=1, keepdims=True))

        beta_coefs = np.asarray(beta_coefs)

        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=0)

        self.x_pred_test = x_test
        self.y_pred = y_pred_list.mean(axis=0)
        self.y_pred_var = y_pred_list.var(axis=0)

        # print("r2:    ", r2_list.mean())
        # print("mse:   ", mse_list.mean())
        # print("bias: ", bias_list.mean())
        # print("var:   ", var_list.mean())


def BootstrapWrapper(x, y, design_matrix, reg, N_bs, test_percent=0.4):
    """
    Wrapper for manual bootstrap method.
    """
    bs_reg = BootstrapRegression(x, y, reg, design_matrix)
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    return {
        "r2": bs_reg.r2, "mse": bs_reg.mse, "bias": bs_reg.bias,
        "var": bs_reg.var, "coef": bs_reg.beta_coefs,
        "coef_var": bs_reg.beta_coefs_var, "x_pred": bs_reg.x_pred_test,
        "y_pred": bs_reg.y_pred, "y_pred_var": bs_reg.y_pred_var}


def SKLearnBootstrap(x, y, design_matrix, reg, N_bs, test_percent=0.4):
    """
    A wrapper for the Scikit-Learn Bootstrap method.
    """
    x_train, x_test, y_train, y_test = sk_modsel.train_test_split(
        x, y, test_size=test_percent, shuffle=False)

    # # Ensures we are on axis shape (N_observations, N_predictors)
    # y_test = y_test.reshape(-1, 1)
    # y_train = y_train.reshape(-1, 1)

    X_test = design_matrix(x_test)
    X_train = design_matrix(x_train)

    # Storage containers for results
    y_pred_array = np.empty((y_test.shape[0], N_bs))
    r2_array = np.empty(N_bs)
    mse_array = np.empty(N_bs)
    # bias_array = np.empty(N_bs)

    beta_coefs = []

    # Using SKLearn to set up indexes
    # bs = sk_cv.Bootstrap(x_train.shape[0], N_bs, n_train=1-test_percent)

    # for i_bs, val_ in enumerate(bs):
    for i_bs in tqdm(range(N_bs), desc="SKLearnBootstrap"):
        # train_index, test_index = val_
        X_boot, y_boot = sk_utils.resample(X_train, y_train)
        # X_boot, y_boot = X_train[train_index], y_train[train_index]

        reg.fit(X_boot, y_boot)
        y_predict = reg.predict(X_test)
        y_pred_array[:, i_bs] = y_predict.ravel()

        r2_array[i_bs] = sk_metrics.r2_score(y_test, y_predict)
        # r2_array[i_bs] = metrics.r2(y_test, y_predict)
        # mse_array[i_bs] = sk_metrics.mean_squared_error(y_test, y_predict)

        beta_coefs.append(reg.coef_)

    # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
    r2 = np.mean(r2_array)

    # # Mean Square Error, mean((y - y_approx)**2)
    # _mse = np.mean((y_test.ravel() - y_pred_list)**2,
    #                axis=0, keepdims=True)
    # mse = np.mean(mse_array)
    _mse = np.mean((y_test - y_pred_array)**2,
                   axis=1, keepdims=True)
    mse = np.mean(_mse)


    # Bias, (y - mean(y_approx))^2
    _y_pred_mean = np.mean(y_pred_array, axis=1, keepdims=True)
    bias = np.mean((y_test - _y_pred_mean)**2)

    # Variance, var(y_approx)
    var = np.mean(np.var(y_pred_array, axis=1, keepdims=True))

    beta_coefs = np.asarray(beta_coefs)

    coef_var = np.asarray(beta_coefs).var(axis=0)
    coef_ = np.asarray(beta_coefs).mean(axis=0)

    x_pred_test = x_test
    y_pred = y_pred_array.mean(axis=1)
    y_pred_var = y_pred_array.var(axis=1)

    return {
        "r2": r2, "mse": mse, "bias": bias,
        "var": var, "coef": coef_,
        "coef_var": coef_var, "x_pred": x_test,
        "y_pred": y_pred, "y_pred_var": y_pred_var}


def __test_bootstrap_fit():
        # A small implementation of a test case
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc

    deg = 2
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)

    N_bs = 1000

    # Initial values
    n = 200
    noise = 0.2
    # np.random.seed(1234)
    test_percent = 0.35

    # Sets up random matrices
    x = np.random.rand(n, 1)

    def func_excact(_x): return 2*_x*_x + np.exp(-2*_x) + noise * \
        np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)

    def design_matrix(_x):
        return poly.fit_transform(_x)

    # Sets up design matrix
    X = design_matrix(x)

    # Performs regression
    reg = OLSRegression()
    reg.fit(X, y)
    y = y.ravel()
    y_predict = reg.predict(X).ravel()
    print("Regular linear regression")
    print("r2:  {:-20.16f}".format(reg.score(y_predict, y)))
    print("mse: {:-20.16f}".format(metrics.mse(y, y_predict)))
    print("Beta:      ", reg.coef_.ravel())
    print("var(Beta): ", reg.coef_var.ravel())
    print("")

    # Performs a bootstrap
    print("Bootstrapping")
    bs_reg = BootstrapRegression(x, y, OLSRegression(), design_matrix)
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    print("r2:    {:-20.16f}".format(bs_reg.r2))
    print("mse:   {:-20.16f}".format(bs_reg.mse))
    print("Bias^2:{:-20.16f}".format(bs_reg.bias))
    print("Var(y):{:-20.16f}".format(bs_reg.var))
    print("Beta:      ", bs_reg.coef_.ravel())
    print("var(Beta): ", bs_reg.coef_var.ravel())
    print("mse = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(bs_reg.mse, bs_reg.bias, bs_reg.var,
                                     bs_reg.bias + bs_reg.var))
    print("Diff: {}".format(abs(bs_reg.bias + bs_reg.var - bs_reg.mse)))

    import matplotlib.pyplot as plt
    plt.plot(x.ravel(), y, "o", label="Data")
    plt.plot(x.ravel(), y_predict, "o",
             label=r"Pred, R^2={:.4f}".format(reg.score(y_predict, y)))
    print(bs_reg.y_pred.shape, bs_reg.y_pred_var.shape)
    plt.errorbar(bs_reg.x_pred_test, bs_reg.y_pred,
                 yerr=np.sqrt(bs_reg.y_pred_var), fmt="o",
                 label=r"Bootstrap Prediction, $R^2={:.4f}$".format(bs_reg.r2))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"$2x^2 + \sigma^2$")
    plt.legend()
    # plt.show()

    # TODO: recreate plot as shown on piazza


def __test_bootstrap():
    import matplotlib.pyplot as plt
    # Data to load and analyse
    data = np.random.normal(0, 2, 100)

    bs_data = np.empty((500, 100))
    # Histogram bins
    N_bins = 20

    # Bootstrapping
    N_bootstraps = int(500)
    for iboot in range(N_bootstraps):
        bs_data[iboot] = np.asarray(boot(data))

    print(data.mean(), data.std())
    bs_data = bs_data.mean(axis=0)
    print(bs_data.mean(), bs_data.std())

    plt.hist(data, label=r"Data, ${0:.3f}\pm{1:.3f}$".format(
        data.mean(), data.std()))
    plt.hist(bs_data, label=r"Bootstrap, ${0:.3f}\pm{1:.3f}$".format(
        bs_data.mean(), bs_data.std()))
    plt.legend()
    plt.show()


def __test_compare_bootstraps():
    # Compare SK learn bootstrap and manual bootstrap
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc
    import copy as cp

    deg = 2
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)

    N_bs = 10000

    # Initial values
    n = 200
    # noise = 0.1
    # np.random.seed(1234)
    test_percent = 0.4

    # Sets up random matrices
    x = np.random.rand(n, 1)

    def func_excact(_x): return 2*_x*_x + np.exp(-2*_x) #+ noise * \
        # np.random.randn(_x.shape[0], _x.shape[1])

    def design_matrix(_x):
        return poly.fit_transform(_x)

    y = func_excact(x)

    bs_my = BootstrapWrapper(
        cp.deepcopy(x), cp.deepcopy(y), design_matrix, OLSRegression(), N_bs, test_percent=test_percent)
    bs_sk = SKLearnBootstrap(
        cp.deepcopy(x), cp.deepcopy(y), design_matrix, OLSRegression(), N_bs, test_percent=test_percent)

    print("r2:", bs_my["r2"], "mse:", bs_my["mse"], "var:", bs_my["var"],
          "bias:", bs_my["bias"], bs_my["mse"] - bs_my["var"] - bs_my["bias"])
    print("r2:", bs_sk["r2"], "mse:", bs_sk["mse"], "var:", bs_sk["var"],
          "bias:", bs_sk["bias"], bs_sk["mse"] - bs_sk["var"] - bs_sk["bias"])


if __name__ == '__main__':
    # __test_bootstrap_fit()
    # __test_bootstrap()
    __test_compare_bootstraps()
