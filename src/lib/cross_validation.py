#!/usr/bin/env python3
import numpy as np
try:
    import lib.metrics as metrics
except ModuleNotFoundError:
    import metrics
from tqdm import tqdm
import sklearn.model_selection as sk_modsel

__all__ = ["kFoldCrossValidation", "MCCrossValidation"]


class __CV_core:
    """Core class for performing k-fold cross validation."""
    _reg = None

    def __init__(self, X_data, y_data, reg):
        """Initializer for Cross Validation.

        Args:
            X_data (ndarray): Design matrix on the shape (N, p)
            y_data (ndarray): y data on the shape (N, 1). Data to be 
                approximated.
            reg (Regression Instance): an initialized regression method
        """
        assert X_data.shape[0] == len(
            y_data), "x and y data not of equal lengths"

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
        """Args:
            rmethod (regression class): regression class to use
        """
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


class kFoldCrossValidation(__CV_core):
    """Class for performing k-fold cross validation."""

    def cross_validate(self, k_splits=5, test_percent=0.2, shuffle=False,
                       X_test=None, y_test=None):
        """
        Args:
            k_splits (float): percentage of the data which is to be used
                for cross validation. Default is 5.
            test_percent (float): size of test data in percent. Optional, 
                default is 0.2.
            X_test (ndarray): design matrix for test values, shape (N, p),
                optional.
            y_test (ndarray): y test data on shape (N, 1), optional.
        """

        N_total_size = self.X_data.shape[0]

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
            y_train = self.y_data

        test_size = y_test.shape[0]

        # Splits kfold train data into k actual folds
        X_subdata = np.array_split(X_train, k_splits, axis=0)
        y_subdata = np.array_split(y_train, k_splits, axis=0)

        # Stores the test values from each k trained data set in an array
        r2_list = np.empty(k_splits)
        beta_coefs = []
        self.y_pred_list = np.empty((test_size, k_splits))

        for ik in tqdm(range(k_splits), desc="k-fold Cross Validation"):

            # Sets up indexes
            set_list = list(range(k_splits))
            set_list.pop(ik)

            # Sets up new data set
            k_X_train = np.concatenate([X_subdata[d] for d in set_list])
            k_y_train = np.concatenate([y_subdata[d] for d in set_list])

            # Trains method bu fitting data
            self.reg.fit(k_X_train, k_y_train)

            # Getting a prediction given the test data
            y_predict = self.reg.predict(X_test).ravel()

            # Appends prediction and beta coefs
            self.y_pred_list[:, ik] = y_predict
            beta_coefs.append(self.reg.coef_)

        # Mean Square Error, mean((y - y_approx)**2)
        _mse = (y_test - self.y_pred_list)**2
        self.mse = np.mean(np.mean(_mse, axis=1, keepdims=True))

        # Bias, (y - mean(y_approx))^2
        _mean_pred = np.mean(self.y_pred_list, axis=1, keepdims=True)
        _bias = y_test - _mean_pred
        self.bias = np.mean(_bias**2)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        _r2 = metrics.r2(y_test, self.y_pred_list, axis=1)
        self.r2 = np.mean(_r2)

        # Variance, var(y_predictions)
        self.var = np.mean(np.var(self.y_pred_list, axis=1, keepdims=True))

        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=1)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=1)

        self.x_pred_test = X_test[:,1]
        self.y_pred = np.mean(self.y_pred_list, axis=1)
        self.y_pred_var = np.var(self.y_pred_list, axis=1)


class kkFoldCrossValidation(__CV_core):
    """A nested k fold CV for getting bias."""

    def cross_validate(self, k_splits=4, test_percent=0.2, X_test=None,
                       y_test=None, shuffle=False):
        """
        Args:
            k_splits (float): Number of k folds to make in the data. Optional,
                default is 4 folds.
            test_percent (float): Percentage of data set to set aside for 
                testing. Optional, default is 0.2.
            X_test (ndarray): design matrix test data, shape (N,p). Optional, 
                default is using 0.2 percent of data as test data.
            y_test (ndarray): design matrix test data. Optional, default is 
                default is using 0.2 percent of data as test data.
        """

        # Checks if we have provided test data or not
        if isinstance(X_test, type(None)) and \
                isinstance(y_test, type(None)):

            # Splits X data and design matrix data
            X_train, X_test, y_train, y_test = \
                sk_modsel.train_test_split(self.X_data, self.y_data,
                                           test_size=test_percent,
                                           shuffle=shuffle)

        else:
            # If X_test and y_test is provided, we simply use those as test 
            # values.
            X_train = self.X_data
            y_train = self.y_data

        N_total_size = X_train.shape[0]

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        holdout_test_size = N_total_size // k_splits

        # In case we have an uneven split
        if (N_total_size % k_splits != 0):
            X_train = X_train[:holdout_test_size*k_splits]
            y_train = y_train[:holdout_test_size*k_splits]

        # Splits data
        X_data = np.split(X_train, k_splits, axis=0)
        y_data = np.split(y_train, k_splits, axis=0)

        # Sets up some arrays for storing the different MSE, bias, var, R^2
        # scores.
        mse_arr = np.empty(k_splits)
        r2_arr = np.empty(k_splits)
        var_arr = np.empty(k_splits)
        bias_arr = np.empty(k_splits)

        beta_coefs = []
        x_pred_test = []
        y_pred_mean_list = []
        y_pred_var_list = []

        for i_holdout in tqdm(range(k_splits),
                              desc="Nested k fold Cross Validation"):

            # Gets the testing holdout data to be used. Makes sure to use
            # every holdout test data once.
            X_holdout = X_data[i_holdout]
            y_holdout = y_data[i_holdout]

            # Sets up indexes
            holdout_set_list = list(range(k_splits))
            holdout_set_list.pop(i_holdout)

            # Sets up new holdout data sets
            X_holdout_train = np.concatenate(
                [X_data[d] for d in holdout_set_list])
            y_holdout_train = np.concatenate(
                [y_data[d] for d in holdout_set_list])

            # Splits dataset into managable k fold tests
            test_size = X_holdout_train.shape[0] // k_splits

            # Splits kfold train data into k actual folds
            X_subdata = np.array_split(X_holdout_train, k_splits, axis=0)
            y_subdata = np.array_split(y_holdout_train, k_splits, axis=0)

            # Stores the test values from each k trained data set in an array
            r2_list = np.empty(k_splits)

            y_pred_list = np.empty((X_test.shape[0], k_splits))

            # Loops over all k-k folds, ensuring every fold is used as a
            # holdout set.
            for ik in range(k_splits):

                # Sets up indexes
                set_list = list(range(k_splits))
                set_list.pop(ik)

                # Sets up new data set
                k_X_train = np.concatenate([X_subdata[d] for d in set_list])
                k_y_train = np.concatenate([y_subdata[d] for d in set_list])

                # Trains method bu fitting data
                self.reg.fit(k_X_train, k_y_train)

                # Appends prediction and beta coefs
                y_pred_list[:, ik] = self.reg.predict(X_test).ravel()
                beta_coefs.append(self.reg.coef_)

            # Mean Square Error, mean((y - y_approx)**2)
            _mse = (y_test - y_pred_list)**2
            mse_arr[i_holdout] = np.mean(np.mean(_mse, axis=1, keepdims=True))

            # Bias, (y - mean(y_approx))^2
            _mean_pred = np.mean(y_pred_list, axis=1, keepdims=True)
            _bias = y_test - _mean_pred
            bias_arr[i_holdout] = np.mean(_bias**2)

            # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
            _r2 = metrics.r2(y_test, y_pred_list, axis=1)
            r2_arr[i_holdout] = np.mean(_r2)

            # Variance, var(y_predictions)
            _var = np.var(y_pred_list, axis=1, keepdims=True)
            var_arr[i_holdout] = np.mean(_var)

            y_pred_mean_list.append(np.mean(y_pred_list, axis=1))
            y_pred_var_list.append(np.var(y_pred_list, axis=1))

        self.var = np.mean(var_arr)
        self.bias = np.mean(bias_arr)
        self.r2 = np.mean(r2_arr)
        self.mse = np.mean(mse_arr)
        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=0)

        self.x_pred_test = X_test[:,1]
        self.y_pred = np.array(y_pred_mean_list).mean(axis=0)
        self.y_pred_var = np.array(y_pred_var_list).mean(axis=0)


class MCCrossValidation(__CV_core):
    """
    https://stats.stackexchange.com/questions/51416/k-fold-vs-monte-carlo-cross-validation
    """

    def cross_validate(self, N_mc, k_splits=4, test_percent=0.2, X_test=None, 
        y_test=None, shuffle=False):
        """
        Args:
            N_mc (int): Number of cross validations to perform
            k_splits (float): Number of k folds to make in the data. Optional,
                default is 4 folds.
            test_percent (float): Percentage of data set to set aside for 
                testing. Optional, default is 0.2.
            X_test (ndarray): Design matrix test data, shape (N,p). Optional, 
                default is using 0.2 percent of data as test data.
            y_test (ndarray): Design matrix test data. Optional, default is 
                default is using 0.2 percent of data as test data.
            shuffle (bool): if True, will shuffle the data before splitting
        """

        # Checks if we have provided test data or not
        if isinstance(X_test, type(None)) and \
                isinstance(y_test, type(None)):

            # Splits X data and design matrix data
            X_train, X_test, y_train, y_test = \
                sk_modsel.train_test_split(self.X_data, self.y_data,
                                           test_size=test_percent,
                                           shuffle=shuffle)

        else:
            # If X_test and y_test is provided, we simply use those as test 
            # values.
            X_train = self.X_data
            y_train = self.y_data

        N_total_size = X_train.shape[0]

        # # Splits X data and design matrix data
        # X_train, X_test, y_train, y_test = \
        #     sk_modsel.train_test_split(self.X_data, self.y_data,
        #                                test_size=test_percent)
        test_size = y_test.shape[0]

        N_mc_data = len(X_train)

        # Splits dataset into managable k fold tests
        mc_test_size = N_mc_data // k_splits

        # All possible indices available
        mc_indices = list(range(N_mc_data))

        # Stores the test values from each k trained data set in an array
        r2_list = np.empty(N_mc)
        beta_coefs = []
        self.y_pred_list = np.empty((test_size, N_mc))

        for i_mc in tqdm(range(N_mc), desc="Monte Carlo Cross Validation"):

            # Gets retrieves indexes for MC-CV. No replacement.
            mccv_test_indexes = np.random.choice(mc_indices, mc_test_size)
            mccv_train_indices = np.array(
                list(set(mc_indices) - set(mccv_test_indexes)))

            # Sets up new data set
            # k_x_train = x_mc_train[mccv_train_indices]
            k_X_train = X_train[mccv_train_indices]
            k_y_train = y_train[mccv_train_indices]

            # Trains method bu fitting data
            self.reg.fit(k_X_train, k_y_train)

            # Adds prediction and beta coefs
            self.y_pred_list[:, i_mc] = self.reg.predict(X_test).ravel()
            beta_coefs.append(self.reg.coef_)

        # Mean Square Error, mean((y - y_approx)**2)
        _mse = (y_test - self.y_pred_list)**2
        self.mse = np.mean(np.mean(_mse, axis=1, keepdims=True))

        # Bias, (y - mean(y_approx))^2
        _mean_pred = np.mean(self.y_pred_list, axis=1, keepdims=True)
        _bias = y_test - _mean_pred
        self.bias = np.mean(_bias**2)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        _r2 = metrics.r2(y_test, self.y_pred_list, axis=0)
        self.r2 = np.mean(_r2)

        # Variance, var(y_predictions)
        self.var = np.mean(np.var(self.y_pred_list, axis=1, keepdims=True))

        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=1)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=1)

        self.x_pred_test = X_test[:,1]
        self.y_pred = np.mean(self.y_pred_list, axis=1)
        self.y_pred_var = np.var(self.y_pred_list, axis=1)


def kFoldCVWrapper(X, y, reg, k=4, test_percent=0.4,
                   shuffle=False, X_test=None, y_test=None):
    """k-fold Cross Validation using a manual method.

    Args:
        X_data (ndarray): design matrix on the shape (N, p)
        y_data (ndarray): y data on the shape (N, 1). Data to be 
            approximated.
        reg (Regression Instance): an initialized regression method
        k (int): optional, number of k folds. Default is 4.
        test_percent (float): optional, size of testing data. Default is 0.4.
        shuffle (bool): optional, if the data will be shuffled. Default is 
            False.
        X_test, (ndarray): design matrix for test values, shape (N, p).
        y_test, (ndarray): y test data on shape (N, 1).

    Return:
        dictionary with r2, mse, bias, var, coef, coef_var
    """

    kfcv_reg = kFoldCrossValidation(X, y, reg)
    kfcv_reg.cross_validate(k_splits=k, test_percent=test_percent,
                            shuffle=shuffle, X_test=X_test, y_test=y_test)

    return {
        "r2": kfcv_reg.r2, "mse": kfcv_reg.mse, "bias": kfcv_reg.bias,
        "var": kfcv_reg.var, 
        "diff": kfcv_reg.mse - kfcv_reg.bias - kfcv_reg.var,
        "coef": kfcv_reg.beta_coefs, "coef_var": kfcv_reg.beta_coefs_var}
    # , "x_pred": kfcv_reg.x_pred_test,
    # "y_pred": kfcv_reg.y_pred, "y_pred_var": kfcv_reg.y_pred_var}


def SKLearnkFoldCV(X, y, reg, k=4, test_percent=0.4,
                   shuffle=False, X_test=None, y_test=None):
    """k-fold Cross Validation using SciKit Learn.

    Args:
        X_data (ndarray): design matrix on the shape (N, p)
        y_data (ndarray): y data on the shape (N, 1). Data to be 
            approximated.
        reg (Regression Instance): an initialized regression method
        k (int): number of k folds. Optional, default is 4.
        test_percent (float): size of testing data. Optional, default is 0.4.
        X_test, (ndarray): design matrix for test values, shape (N, p).
        y_test, (ndarray): y test data on shape (N, 1).

    Return:
        dictionary with r2, mse, bias, var, coef, coef_var
    """

    # kfcv_reg = kFoldCrossValidation(x, y, reg, design_matrix)
    # kfcv_reg.cross_validate(k_splits=k, test_percent=test_percent)

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

    # X_train, X_test, y_train, y_test = sk_modsel.train_test_split(
    #     X, y, test_size=test_percent, shuffle=shuffle)

    # Preps lists to be filled
    y_pred_list = np.empty((y_test.shape[0], k))
    beta_coefs = []

    # Specifies the number of splits
    kfcv = sk_modsel.KFold(n_splits=k, shuffle=shuffle)

    for i, val in tqdm(enumerate(kfcv.split(X_train)), 
        desc="SK-learn k-fold CV"):

        train_index, test_index = val

        reg.fit(X_train[train_index], y_train[train_index])
        y_pred_list[:, i] = reg.predict(X_test).ravel()
        beta_coefs.append(reg.coef_)

    # Mean Square Error, mean((y - y_approx)**2)
    _mse = (y_test - y_pred_list)**2
    mse = np.mean(np.mean(_mse, axis=1, keepdims=True))

    # Bias, (y - mean(y_approx))^2
    _mean_pred = np.mean(y_pred_list, axis=1, keepdims=True)
    _bias = y_test - _mean_pred
    bias = np.mean(_bias**2)

    # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
    r2 = metrics.r2(y_test, y_pred_list, axis=1).mean()

    # Variance, var(y_predictions)
    var = np.mean(np.var(y_pred_list, axis=1, keepdims=True))

    return {"r2": r2, "mse": mse, "bias": bias, "var": var,
            "diff": mse - bias - var,
            "coef": np.asarray(beta_coefs).var(axis=1),
            "coef_var": np.asarray(beta_coefs).mean(axis=1)}

def kkfoldWrapper(X, y, reg, k=4, test_percent=0.4,
                   shuffle=False, X_test=None, y_test=None):
    """k-fold Cross Validation using a manual method.

    Args:
        X_data (ndarray): design matrix on the shape (N, p)
        y_data (ndarray): y data on the shape (N, 1). Data to be 
            approximated.
        reg (Regression Instance): an initialized regression method
        k (int): optional, number of k folds. Default is 4.
        test_percent (float): optional, size of testing data. Default is 0.4.
        shuffle (bool): optional, if the data will be shuffled. Default is 
            False.
        X_test, (ndarray): design matrix for test values, shape (N, p).
        y_test, (ndarray): y test data on shape (N, 1).

    Return:
        dictionary with r2, mse, bias, var, coef, coef_var
    """

    kkfcv_reg = kkFoldCrossValidation(X, y, reg)
    kkfcv_reg.cross_validate(k_splits=k, test_percent=test_percent,
                            shuffle=shuffle, X_test=X_test, y_test=y_test)

    return {
        "r2": kkfcv_reg.r2, "mse": kkfcv_reg.mse, "bias": kkfcv_reg.bias,
        "var": kkfcv_reg.var, 
        "diff": kkfcv_reg.mse - kkfcv_reg.bias - kkfcv_reg.var,
        "coef": kkfcv_reg.beta_coefs, "coef_var": kkfcv_reg.beta_coefs_var}


def MCCVWrapper(X, y, reg, N_mc, k=4, test_percent=0.4, shuffle=False, 
    X_test=None, y_test=None):
    """k-fold Cross Validation using a manual method.

    Args:
        X_data (ndarray): design matrix on the shape (N, p)
        y_data (ndarray): y data on the shape (N, 1). Data to be 
            approximated.
        reg (Regression Instance): an initialized regression method
        N_mc (int): number of MC samples to use.
        k (int): optional, number of k folds. Default is 4.
        test_percent (float): optional, size of testing data. Default is 0.4.
        shuffle (bool): optional, if the data will be shuffled. Default is 
            False.
        X_test, (ndarray): design matrix for test values, shape (N, p).
        y_test, (ndarray): y test data on shape (N, 1).

    Return:
        dictionary with r2, mse, bias, var, coef, coef_var
    """

    mccv_reg = MCCrossValidation(X, y, reg)
    mccv_reg.cross_validate(N_mc, k_splits=k, test_percent=test_percent, 
        X_test=X_test, y_test=y_test, shuffle=shuffle)


def SKLearnMCCV(X, y, reg, N_bs, k=4, test_percent=0.4):
    pass


def __compare_kfold_cv():
    """Runs a comparison between implemented method of k-fold Cross Validation
    and SK-learn's implementation of SK-learn. Since they both are 
    deterministic, should the answer be exactly the same."""
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc
    import sklearn.linear_model as sk_model
    import copy as cp

    deg = 2
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)

    k_splits = 4

    # N_bs = 10000

    # Initial values
    n = 100
    noise = 0.3
    np.random.seed(1234)
    test_percent = 0.35
    shuffle = False

    # Sets up random matrices
    x = np.random.rand(n, 1)
    # x = np.c_[np.linspace(0,1,n)]

    def func_excact(_x):
        return 2*_x*_x + np.exp(-2*_x)  # + noise * \
        #np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)
    X = poly.fit_transform(x)

    kfcv_my = kFoldCVWrapper(
        cp.deepcopy(X), cp.deepcopy(y),
        sk_model.LinearRegression(fit_intercept=False), k=k_splits,
        test_percent=test_percent, shuffle=shuffle)

    print("Manual implementation:")
    print("r2:", kfcv_my["r2"], "mse:", kfcv_my["mse"],
          "var: {:.16f}".format(kfcv_my["var"]),
          "bias: {:.16f}".format(kfcv_my["bias"]),
          "diff: {:.16f}".format(
        abs(kfcv_my["mse"] - kfcv_my["var"] - kfcv_my["bias"])))

    kfcv_sk = SKLearnkFoldCV(
        cp.deepcopy(X), cp.deepcopy(y),
        sk_model.LinearRegression(fit_intercept=False), k=k_splits,
        test_percent=test_percent, shuffle=shuffle)

    print("SK-Learn:")
    print("r2:", kfcv_sk["r2"], "mse:", kfcv_sk["mse"],
          "var: {:.16f}".format(kfcv_sk["var"]),
          "bias: {:.16f}".format(kfcv_sk["bias"]),
          "diff: {:.16f}".format(
        abs(kfcv_sk["mse"] - kfcv_sk["var"] - kfcv_sk["bias"])))


def __compare_mc_cv():
    pass


def __test_cross_validation_methods():
    # A small implementation of a test case
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc
    import matplotlib.pyplot as plt

    # Initial values
    n = 100
    N_bs = 200
    deg = 2
    k_splits = 4
    test_percent = 0.35
    noise = 0.3
    np.random.seed(1234)
    # Sets up random matrices
    x = np.random.rand(n, 1)

    def func_excact(_x): return 2*_x*_x + np.exp(-2*_x) + noise * \
        np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)

    # Sets up design matrix
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(x)

    # Performs regression
    reg = OLSRegression()
    reg.fit(X, y)
    y_predict = reg.predict(X)
    print("Regular linear regression")
    print("R2:    {:-20.16f}".format(reg.score(X, y)))
    print("MSE:   {:-20.16f}".format(metrics.mse(y, y_predict)))
    # print (metrics.bias(y, y_predict))
    print("Bias^2:{:-20.16f}".format(metrics.bias(y, y_predict)))

    # Small plotter
    import matplotlib.pyplot as plt
    plt.plot(x, y, "o", label="data")
    plt.plot(x, y_predict, "o",
             label=r"Pred, $R^2={:.4f}$".format(reg.score(X, y)))

    print("k-fold Cross Validation")
    kfcv = kFoldCrossValidation(X, y, OLSRegression())
    kfcv.cross_validate(k_splits=k_splits,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(kfcv.r2))
    print("MSE:   {:-20.16f}".format(kfcv.mse))
    print("Bias^2:{:-20.16f}".format(kfcv.bias))
    print("Var(y):{:-20.16f}".format(kfcv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kfcv.mse, kfcv.bias, kfcv.var,
                                     kfcv.bias + kfcv.var))
    print("Diff: {}".format(abs(kfcv.bias + kfcv.var - kfcv.mse)))

    plt.errorbar(kfcv.x_pred_test, kfcv.y_pred,
                 yerr=np.sqrt(kfcv.y_pred_var), fmt="o",
                 label=r"k-fold CV, $R^2={:.4f}$".format(kfcv.r2))

    print("kk Cross Validation")
    kkcv = kkFoldCrossValidation(X, y, OLSRegression())
    kkcv.cross_validate(k_splits=k_splits,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(kkcv.r2))
    print("MSE:   {:-20.16f}".format(kkcv.mse))
    print("Bias^2:{:-20.16f}".format(kkcv.bias))
    print("Var(y):{:-20.16f}".format(kkcv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kkcv.mse, kkcv.bias, kkcv.var,
                                     kkcv.bias + kkcv.var))
    print("Diff: {}".format(abs(kkcv.bias + kkcv.var - kkcv.mse)))

    print (kkcv.x_pred_test.shape, kkcv.y_pred.shape, kkcv.y_pred_var.shape)
    plt.errorbar(kkcv.x_pred_test, kkcv.y_pred,
                 yerr=np.sqrt(kkcv.y_pred_var), fmt="o",
                 label=r"kk-fold CV, $R^2={:.4f}$".format(kkcv.r2))

    print("Monte Carlo Cross Validation")
    mccv = MCCrossValidation(X, y, OLSRegression())
    mccv.cross_validate(N_bs, k_splits=k_splits,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(mccv.r2))
    print("MSE:   {:-20.16f}".format(mccv.mse))
    print("Bias^2:{:-20.16f}".format(mccv.bias))
    print("Var(y):{:-20.16f}".format(mccv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(mccv.mse, mccv.bias, mccv.var,
                                     mccv.bias + mccv.var))
    print("Diff: {}".format(abs(mccv.bias + mccv.var - mccv.mse)))

    print("\nCross Validation methods tested.")

    plt.errorbar(mccv.x_pred_test, mccv.y_pred,
                 yerr=np.sqrt(mccv.y_pred_var), fmt="o",
                 label=r"MC CV, $R^2={:.4f}$".format(mccv.r2))

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"$y=2x^2$")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    __test_cross_validation_methods()
    # __compare_kfold_cv()
