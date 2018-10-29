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

    def __init__(self, X_data, y_data, reg, X_test=None, y_test=None):
        """
        Initialises an bootstrap regression object.

        Args:
            X_data (ndarray): Design matrix, on shape (N, p)
            y_data (ndarray): y data, observables, shape (N, 1)
            reg: regression method object. Must have method fit, predict 
                and coef_.
        """

        assert X_data.shape[0] == len(y_data), ("x and y data not of equal"
                                                " lengths")

        assert hasattr(reg, "fit"), ("regression method must have "
                                     "attribute fit()")
        assert hasattr(reg, "predict"), ("regression method must have "
                                         "attribute predict()")

        self.X_data = X_data
        self.y_data = y_data
        self._reg = reg

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
    def bootstrap(self, N_bs, test_percent=0.25, X_test=None, y_test=None):
        """
        Performs a bootstrap for a given regression type, design matrix 
        function and excact function.

        Args:
            N_bs (int): number of bootstraps to perform
            test_percent (float): what percentage of data to reserve for 
                testing. optional, default is 0.25.
            X_test (ndarray): design matrix for test values, shape (N, p), 
                optional. Will use instead of splitting dataset by test 
                percent.
            y_test (ndarray): y test data on shape (N, 1), optional. Will 
                use instead of splitting dataset by test percent.
        """

        assert not isinstance(self._reg, type(None))

        assert test_percent < 1.0, "test_percent must be less than one."

        N = len(self.X_data)

        X = self.X_data
        y = self.y_data

        # Checks if we have provided test data or not
        if isinstance(X_test, type(None)) and \
                isinstance(y_test, type(None)):

            # Splits X data and design matrix data
            X_train, X_test, y_train, y_test = \
                sk_modsel.train_test_split(self.X_data, self.y_data,
                                           test_size=test_percent,
                                           shuffle=False)

        else:
            # If X_test and y_test is provided, we simply use those as test
            # values.
            X_train = self.X_data
            y = self.y_data

        self.x_pred_test = X_test[:, 1]

        test_size = X_test.shape[0]

        # Sets up emtpy lists for gathering the relevant scores in
        r2_list = np.empty(N_bs)
        beta_coefs = []

        y_pred_list = np.empty((test_size, N_bs))

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

        self.y_pred = y_pred_list.mean(axis=1)
        self.y_pred_var = y_pred_list.var(axis=1)


def BootstrapWrapper(X, y, reg, N_bs, test_percent=0.4, X_test=None,
                     y_test=None):
    """
    Wrapper for manual bootstrap method.
    """

    bs_reg = BootstrapRegression(X, y, reg, X_test=X_test, y_test=y_test)
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    return {
        "r2": bs_reg.r2, "mse": bs_reg.mse, "bias": bs_reg.bias,
        "var": bs_reg.var, "diff": bs_reg.mse - bs_reg.bias - bs_reg.var,
        "coef": bs_reg.beta_coefs,
        "coef_var": bs_reg.beta_coefs_var, "x_pred": bs_reg.x_pred_test,
        "y_pred": bs_reg.y_pred, "y_pred_var": bs_reg.y_pred_var}


def SKLearnBootstrap(X, y, reg, N_bs, test_percent=0.4, X_test=None,
                     y_test=None):
    """
    A wrapper for the Scikit-Learn Bootstrap method.
    """

    # Checks if we have provided test data or not
    if ((isinstance(X_test, type(None))) and
            (isinstance(y_test, type(None)))):

        # Splits X data and design matrix data
        X_train, X_test, y_train, y_test = \
            sk_modsel.train_test_split(X, y, test_size=test_percent,
                                       shuffle=False)

    else:
        # If X_test and y_test is provided, we simply use those as test values
        X_train = X
        y = y

    # Storage containers for results
    y_pred_array = np.empty((y_test.shape[0], N_bs))
    r2_array = np.empty(N_bs)
    mse_array = np.empty(N_bs)

    beta_coefs = []

    # for i_bs, val_ in enumerate(bs):
    for i_bs in tqdm(range(N_bs), desc="SKLearnBootstrap"):
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

    X_pred_test = X_test
    y_pred = y_pred_array.mean(axis=1)
    y_pred_var = y_pred_array.var(axis=1)

    return {
        "r2": r2, "mse": mse, "bias": bias,
        "var": var, "diff": mse - bias - var,
        "coef": coef_, "coef_var": coef_var, "x_pred": X_test,
        "y_pred": y_pred, "y_pred_var": y_pred_var}


def __test_bootstrap_fit():
    """A small implementation of a test case."""
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc

    # Initial values
    deg = 2
    N_bs = 1000
    n = 100
    test_percent = 0.35
    noise = 0.3
    np.random.seed(1234)

    # Sets up random matrices
    x = np.random.rand(n, 1)

    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)

    def func_excact(_x): return 2*_x*_x + np.exp(-2*_x) + noise * \
        np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)

    # Sets up design matrix
    X = poly.fit_transform(x)

    # Performs regression
    reg = OLSRegression()
    reg.fit(X, y)
    y_predict = reg.predict(X).ravel()
    print("Regular linear regression")
    print("r2:  {:-20.16f}".format(reg.score(X, y)))
    print("mse: {:-20.16f}".format(metrics.mse(y, reg.predict(X))))
    print("Beta:      ", reg.coef_.ravel())
    print("var(Beta): ", reg.coef_var.ravel())
    print("")

    # Performs a bootstrap
    print("Bootstrapping")
    bs_reg = BootstrapRegression(X, y, OLSRegression())
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
             label=r"Pred, R^2={:.4f}".format(reg.score(X, y)))
    plt.errorbar(bs_reg.x_pred_test, bs_reg.y_pred,
                 yerr=np.sqrt(bs_reg.y_pred_var), fmt="o",
                 label=r"Bootstrap Prediction, $R^2={:.4f}$".format(bs_reg.r2))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"$2x^2 + \sigma^2$")
    plt.legend()
    # plt.show()


# TODO: recreate plot as shown on piazza
def __test_bias_variance_bootstrap():
    """Checks bias-variance relation."""
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc

    # Initial values
    deg_list = np.linspace(1,30,30,dtype=int)
    N_bs = 1000
    n = 100
    test_percent = 0.35
    noise = 0.3
    np.random.seed(1234)

    # Sets up random matrices
    x = np.random.rand(n, 1)

    def func_excact(_x): return 2*_x*_x + np.exp(-2*_x) + noise * \
        np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)

    mse_list = []
    var_list = []
    bias_list = []
    r2_list = []

    for deg in deg_list:
        # Sets up design matrix
        poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
        X = poly.fit_transform(x)

        # Performs regression
        reg = OLSRegression()
        reg.fit(X, y)
        y_predict = reg.predict(X).ravel()

        BootstrapWrapper(X, y, fortset her!!)

        mse_list.append()
        var_list.append()
        bias_list.append()
        r2_list.append()

def __test_basic_bootstrap():
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


def __test_compare_bootstrap_manual_sklearn():
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
    np.random.seed(1234)
    test_percent = 0.4

    # Sets up random matrices
    x = np.random.rand(n, 1)

    def func_excact(_x): return 2*_x*_x + np.exp(-2*_x)  # + noise * \
    # np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)

    X = poly.fit_transform(x)

    bs_my = BootstrapWrapper(
        cp.deepcopy(X), cp.deepcopy(y), OLSRegression(), N_bs,
        test_percent=test_percent)
    bs_sk = SKLearnBootstrap(
        cp.deepcopy(X), cp.deepcopy(y), OLSRegression(), N_bs,
        test_percent=test_percent)

    print("r2:", bs_my["r2"], "mse:", bs_my["mse"], "var:", bs_my["var"],
          "bias:", bs_my["bias"], bs_my["mse"] - bs_my["var"] - bs_my["bias"])
    print("r2:", bs_sk["r2"], "mse:", bs_sk["mse"], "var:", bs_sk["var"],
          "bias:", bs_sk["bias"], bs_sk["mse"] - bs_sk["var"] - bs_sk["bias"])


if __name__ == '__main__':
    # __test_bootstrap_fit()
    __test_bias_variance_bootstrap()
    # __test_basic_bootstrap()
    # __test_compare_bootstrap_manual_sklearn()
