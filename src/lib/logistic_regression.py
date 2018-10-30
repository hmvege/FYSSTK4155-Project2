#!/usr/bin/env python3

import numpy as np
import scipy
import copy as cp

class LogisticRegression:
    """An implementation of Logistic regression."""
    _fit_performed = False
    __possible_backends = ["numpy", "scipy"]
    __possible_inverse_methods = ["inv", "svd"]

    def __init__(self, linalg_backend="scipy", inverse_method="svd"):
        """Sets up the linalg backend."""
        assert linalg_backend in self.__possible_backends, \
            "{:s} backend not recognized".format(str(linalg_backend))
        self.linalg_backend = linalg_backend

        assert inverse_method in self.__possible_inverse_methods, \
            "{:s} inverse method not recognized".format(str(inverse_method))
        self.inverse_method = inverse_method

    @property
    def coef_(self):
        return self.coef

    @coef_.getter
    def coef_(self):
        return self.coef

    @coef_.setter
    def coef_(self, value):
        self.coef = value

    @property
    def coef_var(self):
        return self.beta_coefs_var

    @coef_var.getter
    def coef_var(self):
        return self.beta_coefs_var

    @coef_var.setter
    def coef_var(self, value):
        self.beta_coefs_var = value


    def fit(self, X_train, y_train, solver="gradient_descent"):
        """Performs a linear regression fit for data X_train and y_train."""
        X = cp.deepcopy(X_train)
        y = cp.deepcopy(y_train)

        print (X_train.shape, y_train.shape)

        self.N_features, self.p = X.shape
        _, self.N_labels = y.shape

        self.coef = np.zeros((self.p, self.N_labels))

        Z = X_train @ self.coef

        self._cost_function(X, self.coef, y)
        pass

    def _sigmoid(self, y):
        """Sigmoid function.

        P(y) = 1 / (1 + exp(-y))

        Args:
            y (ndarray): array of predictions.
        Returns:
            (ndarray): probabilities for given y.
        """
        exp_ = np.exp(y)
        return exp_ / (1 + exp_)
        # return 1./(1 + np.exp(y)) # Alternative return

    def _gradient_descent(self):
        pass

    def _cost_function(self, X, beta, y):
        """Cost/loss function for logistic regression. Also known as the 
        cross entropy in statistics.

        Args:
            X (ndarray): design matrix, shape (N, p).
            beta (ndarray): matrix of coefficients (p, labels).
            y (ndarray): predicted values, shape (N, labels).
        Returns:
            (ndarray): 1D array of predictions
        """

        y_pred = self._predict(X, beta)

        p_probabilities = self._sigmoid(y_pred)

        # print (p_probabilities.shape, y_pred.shape)
        cost1 = - y_pred * np.log(p_probabilities)
        cost2 = (1 - y_pred) * np.log(1 - p_probabilities)
        # print (cost1.shape)
        # print (cost2.shape)

        return np.sum(cost1 - cost2)

    def _cost_function_gradient(self, X, beta, y):
        pass

    def _cost_function_laplacian(self, X, beta, y):
        pass

    def _predict(self, X, beta):
        """Performs a regular fit of parameters and sends them through 
        the sigmoid function.

        Args:
            X (ndarray): design matrix/feature matrix, shape (N, p)
            beta (ndarray): coefficients 
        """
        return X @ beta

    def score(self, features_test, labels_test):
        """Returns the mean accuracy of the fit.

        Args:
            features_test (ndarray): array of shape (N, p - 1) to test for
            labels_test (ndarray): true labels

        Returns:
            (float): mean accuracy score for features_test values.
        """

        raise NotImplementedError("score")

        return

    def predict_proba(self, X):
        """Predicts probability of a design matrix X."""


def __test_logistic_regression():
    from sklearn import datasets
    import sklearn.linear_model as sk_model
    import matplotlib.pyplot as plt

    iris = datasets.load_iris()
    X = iris["data"][:, 3:] # petal width
    y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0

    # SK-Learn logistic regression
    sk_log_reg = sk_model.LogisticRegression()
    sk_log_reg.fit(cp.deepcopy(X), cp.deepcopy(y))
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = sk_log_reg.predict_proba(X_new)

    # Manual logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(cp.deepcopy(X), cp.deepcopy(y.reshape(-1,1)))

    fig = plt.figure()

    # SK-Learn logistic regression
    ax1 = fig.add_subplot(211)
    ax1.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
    ax1.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
    ax1.set_title(r"SK-Learn versus manual implementation of Logistic Regression")
    ax1.set_ylabel(r"Probability")
    ax1.legend()

    # Manual logistic regression
    ax2 = fig.add_subplot(212)
    ax2.set_ylabel(r"Probability")

    # ax2.legend()
    # plt.show()


if __name__ == '__main__':
    __test_logistic_regression()
