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
    _design_matrix = None

    def __init__(self, x_data, y_data, reg, design_matrix_func):
        """Initializer for Cross Validation.

        Args:
            x_data (ndarray): x data on the shape (N, 1)
            y_data (ndarray): y data on the shape (N, 1). Data to be 
                approximated.
            reg (Regression Instance): an initialized regression method
            design_matrix_func (function): function that sets up the design
                matrix.
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

    def cross_validate(self, k_splits=5, test_percent=0.2, shuffle=False):
        """
        Args:
            k_splits (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """

        N_total_size = self.x_data.shape[0]

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        # training_size = int(np.floor(N_total_size * (1 - test_percent))) # Last part of data set is set of for testing

        # if shuffle:
        #     np.random.shuffle(self.x_data)
        #     np.random.shuffle(self.y_data)

        # # Manual splitting
        # x_test = self.x_data[training_size:, :]
        # x_train = self.x_data[:training_size, :]
        # y_test = self.y_data[training_size:]
        # y_train = self.y_data[:training_size]

        # print(x_test.shape, y_test.shape, x_train.shape, y_train.shape, training_size)

        x_train, x_test, y_train, y_test = \
            sk_modsel.train_test_split(self.x_data, self.y_data,
                                       test_size=test_percent, shuffle=shuffle)

        # if shuffle:
        #     np.random.shuffle(x_test)
        #     np.random.shuffle(y_test)
        #     np.random.shuffle(x_train)
        #     np.random.shuffle(y_train)

        # print(x_test.shape, y_test.shape, x_train.shape, y_train.shape, training_size)

        test_size = y_test.shape[0]

        # Sets up the holdout design matrix
        X_test = self._design_matrix(x_test)
        X_train = self._design_matrix(x_train)

        # Splits dataset into managable k fold tests
        # N_kfold_data = len(y_train)
        # test_size = int(np.floor(N_kfold_data / k_splits))

        # Splits kfold train data into k actual folds
        X_subdata = np.array_split(X_train, k_splits, axis=0)
        y_subdata = np.array_split(y_train, k_splits, axis=0)

        # Stores the test values from each k trained data set in an array
        r2_list = np.empty(k_splits)
        beta_coefs = []
        self.y_pred_list = np.empty((test_size, k_splits))

        for ik in tqdm(range(k_splits), desc="k-fold Cross Validation"):
            # Gets the testing data
            # k_x_test = x_subdata[ik]
            # k_y_test = y_subdata[ik]

            # X_test = self._design_matrix(k_x_test)

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

        self.x_pred_test = x_test
        self.y_pred = np.mean(self.y_pred_list, axis=1)
        self.y_pred_var = np.var(self.y_pred_list, axis=1)


class kkFoldCrossValidation(__CV_core):
    """A nested k fold CV for getting bias."""

    def cross_validate(self, k_splits=4, kk_splits=4, test_percent=0.2):
        """
        Args:
            k_splits (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """
        # raise NotImplementedError("Not implemnted kk fold CV")

        N_total_size = len(self.x_data)

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        holdout_test_size = int(np.floor(N_total_size/k_splits))

        x_holdout_data = np.split(self.x_data, k_splits, axis=0)
        y_holdout_data = np.split(self.y_data, k_splits, axis=0)

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
            x_holdout_test = x_holdout_data[i_holdout]
            y_holdout_test = y_holdout_data[i_holdout]

            # Sets up indexes
            holdout_set_list = list(range(k_splits))
            holdout_set_list.pop(i_holdout)

            # Sets up new holdout data sets
            x_holdout_train = np.concatenate(
                [x_holdout_data[d] for d in holdout_set_list])
            y_holdout_train = np.concatenate(
                [y_holdout_data[d] for d in holdout_set_list])

            # Sets up the holdout design matrix
            X_holdout_test = self._design_matrix(x_holdout_test)

            # Splits dataset into managable k fold tests
            N_holdout_data = len(x_holdout_train)
            test_size = int(np.floor(N_holdout_data/kk_splits))

            # Splits kfold train data into k actual folds
            x_subdata = np.array_split(x_holdout_train, kk_splits, axis=0)
            y_subdata = np.array_split(y_holdout_train, kk_splits, axis=0)
            X_subdata = self._design_matrix(x_subdata)

            # Stores the test values from each k trained data set in an array
            r2_list = np.empty(kk_splits)

            self.y_pred_list = np.empty((holdout_test_size, kk_splits))
            # self.y_test_list = np.empty((kk_splits, holdout_test_size))

            for ik in range(kk_splits):
                # Gets the testing data
                # k_x_test = x_subdata[ik]
                # k_y_test = y_subdata[ik]

                # X_test = self._design_matrix(k_x_test)

                # Sets up indexes
                set_list = list(range(kk_splits))
                set_list.pop(ik)

                # Sets up new data set
                k_X_train = np.concatenate([X_subdata[d] for d in set_list])
                k_y_train = np.concatenate([y_subdata[d] for d in set_list])

                # # Sets up function to predict
                # X_train = self._design_matrix(k_x_train)

                # Trains method bu fitting data
                self.reg.fit(k_X_train, k_y_train)

                # Appends prediction and beta coefs
                self.y_pred_list[:, ik] = self.reg.predict(
                    X_holdout_test).ravel()
                beta_coefs.append(self.reg.coef_)

            # Mean Square Error, mean((y - y_approx)**2)
            _mse = (y_holdout_test - self.y_pred_list)**2
            mse_arr[i_holdout] = np.mean(np.mean(_mse, axis=1, keepdims=True))

            # Bias, (y - mean(y_approx))^2
            _mean_pred = np.mean(self.y_pred_list, axis=1, keepdims=True)
            _bias = y_holdout_test - _mean_pred
            bias_arr[i_holdout] = np.mean(_bias**2)

            # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
            _r2 = metrics.r2(y_holdout_test, self.y_pred_list, axis=1)
            r2_arr[i_holdout] = np.mean(_r2)

            # Variance, var(y_predictions)
            _var = np.var(self.y_pred_list, axis=1, keepdims=True)
            var_arr[i_holdout] = np.mean(_var)

            x_pred_test.append(x_holdout_test)
            y_pred_mean_list.append(np.mean(self.y_pred_list, axis=1))
            y_pred_var_list.append(np.var(self.y_pred_list, axis=1))

        self.var = np.mean(var_arr)
        self.bias = np.mean(bias_arr)
        self.r2 = np.mean(r2_arr)
        self.mse = np.mean(mse_arr)
        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=0)

        self.x_pred_test = np.array(x_pred_test)
        self.y_pred = np.array(y_pred_mean_list)
        self.y_pred_var = np.array(y_pred_var_list)


class MCCrossValidation(__CV_core):
    """
    https://stats.stackexchange.com/questions/51416/k-fold-vs-monte-carlo-cross-validation
    """

    def cross_validate(self, N_mc_crossvalidations, k_splits=4,
                       test_percent=0.2):
        """
        Args:
            k_splits (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """
        # raise NotImplementedError("Not implemnted MC CV")

        N_total_size = len(self.x_data)

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        # k_holdout, holdout_test_size = self._get_split_percent(
        #     test_percent, N_total_size, enforce_equal_intervals=False)

        # # Splits X data and design matrix data
        # x_holdout_test, x_mc_train = np.split(self.x_data,
        #                                       [holdout_test_size], axis=0)
        # y_holdout_test, y_mc_train = np.split(self.y_data,
        #                                       [holdout_test_size], axis=0)

        # Splits X data and design matrix data
        x_mc_train, x_holdout_test, y_mc_train, y_holdout_test = \
            sk_modsel.train_test_split(self.x_data, self.y_data,
                                       test_size=test_percent)
        holdout_test_size = y_holdout_test.shape[0]

        N_mc_data = len(x_mc_train)

        # Sets up the holdout design matrix
        X_holdout_test = self._design_matrix(x_holdout_test)

        # Splits dataset into managable k fold tests
        mc_test_size = int(np.floor(N_mc_data / k_splits))

        # Splits kfold train data into k actual folds
        # x_subdata = np.array_split(x_kfold_train, k_splits, axis=0)
        # y_subdata = np.array_split(y_kfold_train, k_splits, axis=0)

        # All possible indices available
        mc_indices = list(range(N_mc_data))

        # Stores the test values from each k trained data set in an array
        r2_list = np.empty(N_mc_crossvalidations)
        beta_coefs = []
        self.y_pred_list = np.empty((holdout_test_size, N_mc_crossvalidations))

        # Sets up design matrices beforehand
        X_mc_train = self._design_matrix(x_mc_train)

        for i_mc in tqdm(range(N_mc_crossvalidations),
                         desc="Monte Carlo Cross Validation"):

            # Gets retrieves indexes for MC-CV. No replacement.
            mccv_test_indexes = np.random.choice(mc_indices, mc_test_size)
            mccv_train_indices = np.array(
                list(set(mc_indices) - set(mccv_test_indexes)))

            # # Gets the testing data
            # k_x_test = x_mc_train[mccv_test_indexes]
            # k_y_test = y_mc_train[mccv_test_indexes]

            # X_test = self._design_matrix(k_x_test)

            # # Sets up indexes
            # set_list = list(range(k_splits))
            # set_list.pop(ik)

            # Sets up new data set
            # k_x_train = x_mc_train[mccv_train_indices]
            X_train = X_mc_train[mccv_train_indices]
            k_y_train = y_mc_train[mccv_train_indices]

            # Sets up function to predict
            # X_train = self._design_matrix(k_x_train)

            # Trains method bu fitting data
            self.reg.fit(X_train, k_y_train)

            # Getting a prediction given the test data
            y_predict = self.reg.predict(X_holdout_test).ravel()

            # Appends prediction and beta coefs
            self.y_pred_list[:, i_mc] = y_predict
            beta_coefs.append(self.reg.coef_)

        # Mean Square Error, mean((y - y_approx)**2)
        _mse = (y_holdout_test - self.y_pred_list)**2
        self.mse = np.mean(np.mean(_mse, axis=1, keepdims=True))

        # Bias, (y - mean(y_approx))^2
        _mean_pred = np.mean(self.y_pred_list, axis=1, keepdims=True)
        _bias = y_holdout_test - _mean_pred
        self.bias = np.mean(_bias**2)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        _r2 = metrics.r2(y_holdout_test, self.y_pred_list, axis=1)
        self.r2 = np.mean(_r2)

        # Variance, var(y_predictions)
        self.var = np.mean(np.var(self.y_pred_list, axis=1, keepdims=True))

        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=1)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=1)

        self.x_pred_test = x_holdout_test
        self.y_pred = np.mean(self.y_pred_list, axis=1)
        self.y_pred_var = np.var(self.y_pred_list, axis=1)


def kFoldCVWrapper(x, y, design_matrix, reg, k=4, test_percent=0.4, 
    shuffle=False):
    """k-fold Cross Validation using a manual method.

    Args:
        x_data (ndarray): x data on the shape (N, 1)
        y_data (ndarray): y data on the shape (N, 1). Data to be 
            approximated.
        design_matrix_func (function): function that sets up the design
            matrix.
        reg (Regression Instance): an initialized regression method
        k (int): optional, number of k folds. Default is 4.
        test_percent (float): optional, size of testing data. Default is 0.4.
        shuffle (bool): optional, if the data will be shuffled. Default is 
            False.

    Return:
        dictionary with r2, mse, bias, var, coef, coef_var
    """

    kfcv_reg = kFoldCrossValidation(x, y, reg, design_matrix)
    kfcv_reg.cross_validate(k_splits=k, test_percent=test_percent, shuffle=shuffle)

    return {
        "r2": kfcv_reg.r2, "mse": kfcv_reg.mse, "bias": kfcv_reg.bias,
        "var": kfcv_reg.var, "coef": kfcv_reg.beta_coefs,
        "coef_var": kfcv_reg.beta_coefs_var}
        # , "x_pred": kfcv_reg.x_pred_test,
        # "y_pred": kfcv_reg.y_pred, "y_pred_var": kfcv_reg.y_pred_var}


def SKLearnkFoldCV(x, y, design_matrix, reg, k=4, test_percent=0.4, 
    shuffle=False):
    """k-fold Cross Validation using SciKit Learn.

    Args:
        x_data (ndarray): x data on the shape (N, 1)
        y_data (ndarray): y data on the shape (N, 1). Data to be 
            approximated.
        design_matrix_func (function): function that sets up the design
            matrix.
        reg (Regression Instance): an initialized regression method
        k (int): number of k folds. Optional, default is 4.
        test_percent (float): size of testing data. Optional, default is 0.4.

    Return:
        dictionary with r2, mse, bias, var, coef, coef_var
    """

    # kfcv_reg = kFoldCrossValidation(x, y, reg, design_matrix)
    # kfcv_reg.cross_validate(k_splits=k, test_percent=test_percent)

    x_train, x_test, y_train, y_test = sk_modsel.train_test_split(
        x, y, test_size=test_percent, shuffle=shuffle)
    
    # Sets up the design matrices
    X_test = design_matrix(x_test)
    X_train = design_matrix(x_train)

    # Preps lists to be filled    
    y_pred_list = np.empty((y_test.shape[0], k))
    beta_coefs = []

    # Specifies the number of splits
    kfcv = sk_modsel.KFold(n_splits=k, shuffle=shuffle)

    for i, val in tqdm(enumerate(kfcv.split(X_train)), desc="SK-learn k-fold CV"):
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
            "coef": np.asarray(beta_coefs).var(axis=1), 
            "coef_var": np.asarray(beta_coefs).mean(axis=1)}


def MCCVWrapper(x, y, design_matrix, reg, N_bs, k=4, test_percent=0.4):
    pass


def SKLearnMCCV(x, y, design_matrix, reg, N_bs, k=4, test_percent=0.4):
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
    n = 200
    noise = 0.0
    np.random.seed(1234)
    test_percent = 0.4
    shuffle = False

    # Sets up random matrices
    x = np.random.rand(n, 1)
    # x = np.c_[np.linspace(0,1,n)]

    def func_excact(_x):
        return 2*_x*_x + np.exp(-2*_x) #+ noise * \
            #np.random.randn(_x.shape[0], _x.shape[1])

    def design_matrix(_x):
        return poly.fit_transform(_x)

    y = func_excact(x)

    kfcv_my = kFoldCVWrapper(
        cp.deepcopy(x), cp.deepcopy(y), poly.fit_transform,
        sk_model.LinearRegression(fit_intercept=False), k=k_splits,
        test_percent=test_percent, shuffle=shuffle)

    print("Manual implementation:")
    print("r2:", kfcv_my["r2"], "mse:", kfcv_my["mse"],
          "var: {:.16f}".format(kfcv_my["var"]),
          "bias: {:.16f}".format(kfcv_my["bias"]),
          "diff: {:.16f}".format(
        abs(kfcv_my["mse"] - kfcv_my["var"] - kfcv_my["bias"])))

    kfcv_sk = SKLearnkFoldCV(
        cp.deepcopy(x), cp.deepcopy(y), design_matrix,
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
    import matplotlib.pyplot as plt

    # Initial values
    n = 100
    N_bs = 1000
    k_splits = 4
    test_percent = 0.2
    noise = 0.3
    np.random.seed(1234)

    # Sets up random matrices
    x = np.random.rand(n, 1)

    def func_excact(_x): return 2*_x*_x + np.exp(-2*_x) + noise * \
        np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)

    def design_matrix(_x):
        return np.c_[np.ones(_x.shape), _x, _x*_x]

    # Sets up design matrix
    X = design_matrix(x)

    # Performs regression
    reg = OLSRegression()
    reg.fit(X, y)
    y_predict = reg.predict(X)
    print("Regular linear regression")
    print("R2:    {:-20.16f}".format(reg.score(y, y_predict)))
    print("MSE:   {:-20.16f}".format(metrics.mse(y, y_predict)))
    # print (metrics.bias(y, y_predict))
    print("Bias^2:{:-20.16f}".format(metrics.bias(y, y_predict)))

    # Small plotter
    import matplotlib.pyplot as plt
    plt.plot(x, y, "o", label="data")
    plt.plot(x, y_predict, "o",
             label=r"Pred, $R^2={:.4f}$".format(reg.score(y, y_predict)))

    print("k-fold Cross Validation")
    kfcv = kFoldCrossValidation(x, y, OLSRegression(), design_matrix)
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

    # plt.errorbar(kfcv.x_pred_test, kfcv.y_pred,
    #              yerr=np.sqrt(kfcv.y_pred_var), fmt="o",
    #              label=r"k-fold CV, $R^2={:.4f}$".format(kfcv.r2))

    print("kk Cross Validation")
    kkcv = kkFoldCrossValidation(x, y, OLSRegression(), design_matrix)
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

    # print (kkcv.x_pred_test.shape, kkcv.y_pred.shape, kkcv.y_pred_var.shape)
    # plt.errorbar(kkcv.x_pred_test.ravel(), kkcv.y_pred.ravel(),
    #              yerr=np.sqrt(kkcv.y_pred_var.ravel()), fmt="o",
    #              label=r"kk-fold CV, $R^2={:.4f}$".format(kkcv.r2))

    print("Monte Carlo Cross Validation")
    mccv = MCCrossValidation(x, y, OLSRegression(), design_matrix)
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

    # plt.errorbar(mccv.x_pred_test, mccv.y_pred,
    #              yerr=np.sqrt(mccv.y_pred_var), fmt="o",
    #              label=r"MC CV, $R^2={:.4f}$".format(mccv.r2))

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"$y=2x^2$")
    plt.legend()
    # plt.show()


# TODO: double check that my CV gives the same as the built in CV


if __name__ == '__main__':
    # __test_cross_validation_methods()
    __compare_kfold_cv()
