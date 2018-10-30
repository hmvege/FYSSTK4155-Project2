#!/usr/bin/env python3

import numpy as np
import scipy


class LogisticRegression:
    """Backend class in case we want to run with either scipy, numpy 
    (or something else)."""
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

    def fit(self, X_train, y_train):
        pass

    def _inv(self, M):
        """Method for taking derivatives with either numpy or scipy."""

        if self.linalg_backend == "numpy":

            if self.inverse_method == "inv":
                return np.linalg.inv(M)

            elif self.inverse_method == "svd":
                U, S, VH = np.linalg.svd(M)
                S = np.diag(1.0/S)
                return U @ S @ VH

        elif self.linalg_backend == "scipy":

            if self.inverse_method == "inv":
                return scipy.linalg.inv(M)

            elif self.inverse_method == "svd":
                U, S, VH = scipy.linalg.svd(M)
                S = np.diag(1.0/S)
                return U @ S @ VH

    def score(self, X, y_true):
        """Returns the R^2 score.

        Args:
            X (ndarray): X array of shape (N, p - 1) to test for
            y_true (ndarray): true values for X

        Returns:
            float: R2 score for X_test values.
        """
        return metrics.r2(y_true, self.predict(X))


def __test_logistic_regression():
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression 
    import matplotlib.pyplot as plt

    iris = datasets.load_iris()
    X = iris["data"][:, 3:] # petal width
    y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)
    plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
    plt.show()


if __name__ == '__main__':
    __test_logistic_regression()
