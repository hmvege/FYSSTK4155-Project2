#!/usr/bin/env python3

import numpy as np
import scipy
import copy as cp


def _l1(weights):
    return np.linalg.norm(weights, ord=1)


def _l1_derivative(weights):
    # NOTE: include this in report
    # https://math.stackexchange.com/questions/141101/minimizing-l-1-regularization
    return np.sign(weights)


def _l2(weights):
    return np.linalg.norm(weights)


def _l2_derivative(weights):
    # NOTE: include this in report
    # https://math.stackexchange.com/questions/2792390/derivative-of-euclidean-norm-l2-norm
    return 2*weights


class _OptimizerBase:
    """Base class for optimization."""

    def set_regularization_method(self, penalty):
        """Set the penalty/regularization method to use."""

        self.penalty = penalty

        if penalty == "l1":
            self._get_penalty = lambda x: 0.0
        elif penalty == "l2":
            self._get_penalty = lambda x: 0.0
        elif penalty == None:
            self._get_penalty = lambda x: 0.0
        else:
            raise KeyError(("{} not recognized as a regularization"
                            " method.".format(penalty)))

    def optimize(self):
        pass


class GradientDescent(_OptimizerBase):
    pass


class ConjugateGradient(_OptimizerBase):
    pass


class SGA(_OptimizerBase):
    pass


class NewtonRaphson(_OptimizerBase):
    pass


class LogisticRegression:
    """An implementation of Logistic regression."""
    _fit_performed = False

    def __init__(self, solver="gradient_descent", max_iter=100,
                 penalty="l2", tol=1e-4, lr=1.0, lmbda=1.0):
        """Sets up the linalg backend.

        Args:
            solver (str): what kind of solver method to use. Default is 
                'gradient_descent'.
            max_iter (int): number of iterations to run gradient descent for,
                default is 100.
            penalty (str): what kind of regulizer to use, either 'l1' or 'l2'. 
                Optional, default is 'l2'.
            tol (float): tolerance or when to cut of calculations. Optional, 
                default is 1e-4.
            lr (float): learning reate. Optional, default is 1.0.
            lmbda (float): regularization strength. Default is 1.0.
        """

        self._set_optimizer(solver)
        self._set_regularization_method(penalty)
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.lmbda = lmbda

    def _set_optimizer(self, solver):
        """Set the penalty/regularization method to use."""
        self.solver = solver

        if solver == "gradient_descent":  # aka Steepest descent
            self._get_solver = None
        elif solver == "conjugate_gradient":
            self._get_solver = None
        elif solver == "sga":  # Stochastic Gradient Descent
            self._get_solver = None
        elif solver == "nr":  # Newton-Raphson method
            self._get_solver = None
        else:
            raise KeyError(("{} not recognized as a solver"
                            " method.".format(solver)))

    def _set_regularization_method(self, penalty):
        """Set the penalty/regularization method to use."""

        self.penalty = penalty

        if penalty == "l1":
            self._get_penalty = _l1
            self._get_penalty_derivative = _l1_derivative
        elif penalty == "l2":
            self._get_penalty = _l2
            self._get_penalty_derivative = _l2_derivative
        elif isinstance(type(penalty), None):
            self._get_penalty = lambda x: 0.0
            self._get_penalty_derivative = lambda x: 0.0
        else:
            raise KeyError(("{} not recognized as a regularization"
                            " method.".format(penalty)))

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

    def fit(self, X_train, y_train, alpha=1.0):
        """Performs a linear regression fit for data X_train and y_train.

        Args:
            X_train (ndarray):
            y_train (ndarray):
        """

        X = cp.deepcopy(X_train)
        y = cp.deepcopy(y_train)

        self.N_features, self.p = X.shape
        _, self.N_labels = y.shape

        # Adds constant term and increments the number of predictors
        X = np.hstack([np.ones((self.N_features, self.p)), X])
        self.p += 1

        # Adds beta_0 coefficients
        self.coef = np.zeros((self.p, self.N_labels))
        self.coef[0, :] = np.ones(self.N_labels)

        # Sets up method for storing cost function values
        self.cost_values = []
        self.cost_values.append(self._cost_function(X, y, self.coef))

        # Temp, clean this part up
        def learning_rate(t, t0, t1):
            return t0 / (t + t1)

        # Temp, sets the regularization parameter
        self.alpha = alpha

        for i in range(self.max_iter):
            # Calls the optimizer class which
            # self.coef = self._optimize(X, y, self.coef)

            # # OLD
            self.coef = self._gradient_descent(X, y, self.coef, self.lr)
            # # self.coef += self._l2_regularization(self.coef)

            # Appends cost function values
            self.cost_values.append(self._cost_function(X, y, self.coef))

        # self.coef[0, 0] = 2.61789264

        self._fit_performed = True

    def _l2_regularization(self, weights):
        return np.min(0.5*(weights.T @ weights))

    def _l1_regularization(self, weights):
        return np.min(np.linalg.norm(weights, ord=1))

    def _gradient_descent(self, X, y, weights, lr=1.0):
        """Cost/loss function for logistic regression. Also known as the 
        cross entropy in statistics.

        Args:
            X (ndarray): design matrix, shape (N, p).
            y (ndarray): predicted values, shape (N, labels).
            weights (ndarray): matrix of coefficients (p, labels).
            lr (float): learning rate. Default is 0.1.
        Returns:
            (ndarray): 1D array of weights
        """

        # TODO: move this to outside? Make it take cost_function_gradient and features

        # y_pred = self._predict(X, weights)

        gradient = self._cost_function_gradient(
            X, y, weights) / X.shape[0]
        weights -= gradient*lr
        return weights

    def _cost_function(self, X, y, weights):
        """Cost/loss function for logistic regression. Also known as the 
        cross entropy in statistics.

        Args:
            X (ndarray): design matrix, shape (N, p).
            y (ndarray): predicted values, shape (N, labels).
            weights (ndarray): matrix of coefficients (p, labels).
        Returns:
            (ndarray): 1D array of predictions
        """

        y_pred = self._predict(X, weights)

        p_probabilities = self._sigmoid(y_pred)

        # print (p_probabilities.shape, y_pred.shape)
        cost1 = - y * np.log(p_probabilities)
        cost2 = (1 - y) * np.log(1 - p_probabilities)

        cost = np.sum(cost1 - cost2) + self._get_penalty(weights)

        return cost

    def _cost_function_gradient(self, X, y, weights):
        """Takes the gradient of the cost function w.r.t. the coefficients.

            dC(W)/dw = - X^T * (y - p(X^T * w))
        """
        grad = X.T @ (self._sigmoid(self._predict(X, weights)) - y)
        grad += self.alpha*self._get_penalty_derivative(weights)
        return grad

    def _cost_function_laplacian(self, X, y, w):
        """Takes the laplacian of the cost function w.r.t. the coefficients.

            d^2C(w) / (w w^T) = X^T W X
        where
            W = p(1 - X^T * w) * p(X^T * w)
        """
        y_pred = self._predict(X, w)
        return X.T @ self._sigmoid(1 - y_pred) @ self._sigmoid(y_pred) @ X

    def _sigmoid(self, y):
        """Sigmoid function.

        P(y) = 1 / (1 + exp(-y))

        Args:
            y (ndarray): array of predictions.
        Returns:
            (ndarray): probabilities for given y.
        """

        # exp_ = np.exp(y)
        # return exp_ / (1 + exp_)

        return 1./(1 + np.exp(-y))  # Alternative return

    def _predict(self, X, weights):
        """Performs a regular fit of parameters and sends them through 
        the sigmoid function.

        Args:
            X (ndarray): design matrix/feature matrix, shape (N, p)
            weights (ndarray): coefficients 
        """
        return X @ weights

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

        if not self._fit_performed:
            raise UserWarning("Fit not performed.")

        X = np.hstack([np.ones(X.shape), X])
        probabilities = self._sigmoid(self._predict(X, self.coef))
        results = np.asarray([1 - probabilities, probabilities])
        return np.moveaxis(results, 0, 1)


def __test_logistic_regression():
    from sklearn import datasets
    import sklearn.linear_model as sk_model
    import matplotlib.pyplot as plt

    iris = datasets.load_iris()
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

    # SK-Learn logistic regression
    sk_log_reg = sk_model.LogisticRegression(
        solver="liblinear", C=1.0, penalty="l2", max_iter=10000)
    sk_log_reg.fit(cp.deepcopy(X), cp.deepcopy(y))
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_sk_proba = sk_log_reg.predict_proba(X_new)

    print("SK-learn coefs: ", sk_log_reg.intercept_, sk_log_reg.coef_)

    # Manual logistic regression
    log_reg = LogisticRegression(penalty="l1", lr=1.0, max_iter=100000)
    log_reg.fit(cp.deepcopy(X), cp.deepcopy(y.reshape(-1, 1)))
    y_proba = log_reg.predict_proba(X_new)

    print("Manual coefs:", log_reg.coef_)

    fig = plt.figure()

    # SK-Learn logistic regression
    ax1 = fig.add_subplot(211)
    ax1.plot(X_new, y_sk_proba[:, 1], "g-", label="Iris-Virginica(SK-Learn)")
    ax1.plot(X_new, y_sk_proba[:, 0], "b--",
             label="Not Iris-Virginica(SK-Learn)")
    ax1.set_title(
        r"SK-Learn versus manual implementation of Logistic Regression")
    ax1.set_ylabel(r"Probability")
    ax1.legend()

    # Manual logistic regression
    ax2 = fig.add_subplot(212)
    ax2.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica(Manual)")
    ax2.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica(Manual)")
    ax2.set_ylabel(r"Probability")
    ax2.legend()
    plt.show()


if __name__ == '__main__':
    __test_logistic_regression()
