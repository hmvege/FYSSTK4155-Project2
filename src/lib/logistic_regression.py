#!/usr/bin/env python3

import numpy as np
import scipy
import copy as cp
import abc
import utils.math_tools as umath
from utils.math_tools import AVAILABLE_OUTPUT_ACTIVATIONS


def _l1(weights):
    """The L1 norm."""
    return np.linalg.norm(weights, ord=1)


def _l1_derivative(weights):
    """The derivative of the L1 norm."""
    # NOTE: Include this in report
    # https://math.stackexchange.com/questions/141101/minimizing-l-1-regularization
    return np.sign(weights)


def _l2(weights):
    """The L2 norm."""
    return np.linalg.norm(weights)


def _l2_derivative(weights):
    """The derivative of the L2 norm."""
    # NOTE: Include this in report
    # https://math.stackexchange.com/questions/2792390/derivative-of-
    # euclidean-norm-l2-norm
    return 2*weights


class _OptimizerBase(abc.ABC):
    """Base class for optimization."""

    def __init__(self):
        """No initialization needed."""

    # def set_regularization_method(self, penalty):
    #     """Set the penalty/regularization method to use."""

    #     self.penalty = penalty

    #     if penalty == "l1":
    #         self._get_penalty = lambda x: 0.0
    #     elif penalty == "l2":
    #         self._get_penalty = lambda x: 0.0
    #     elif penalty == None:
    #         self._get_penalty = lambda x: 0.0
    #     else:
    #         raise KeyError(("{} not recognized as a regularization"
    #                         " method.".format(penalty)))

    # Abstract class methods makes it so that thet MUST be overwritten
    @abc.abstractmethod
    def optimize(self, f, x0):
        pass


class GradientDescent(_OptimizerBase):
    pass


class ConjugateGradient(_OptimizerBase):
    def optimize(self, f, x0):
        return scipy.optimize(f, x0, method="CG")


class SGA(_OptimizerBase):
    pass


class NewtonRaphson(_OptimizerBase):
    def optimize(self, f, x0):
        return scipy.optimize.newton(f, x0)


class LogisticRegression:
    """An implementation of Logistic regression."""
    _fit_performed = False

    def __init__(self, solver="gradient_descent", activation="sigmoid",
                 max_iter=100, penalty="l2", tol=1e-4, lr=1.0, alpha=1.0,
                 momentum=0.0):
        """Sets up the linalg backend.

        Args:
            solver (str): what kind of solver method to use. Default is 
                'gradient_descent'.
            activation (str): type of activation function to use. Optional, 
                default is 'sigmoid'.
            max_iter (int): number of iterations to run gradient descent for,
                default is 100.
            penalty (str): what kind of regulizer to use, either 'l1' or 'l2'. 
                Optional, default is 'l2'.
            tol (float): tolerance or when to cut of calculations. Optional, 
                default is 1e-4.
            alpha (float): regularization strength. Default is 1.0.
            momentum (float): adds a momentum, in which the current gradient 
                deepends on the last gradient. Default is 0.0.
        """

        self._set_optimizer(solver)
        self._set_activation_function(activation)
        self._set_regularization_method(penalty)

        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.momentum = momentum

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

    def _set_learning_rate(self, eta):
        """Sets the learning rate."""
        if isinstance(eta, float):
            self._update_learning_rate = lambda _i, _N: eta
        elif eta == "inverse":
            self._update_learning_rate = lambda _i, _N: 1 - _i/float(_N+1)
        else:
            raise KeyError(("Eta {} is not recognized learning"
                            " rate.".format(eta)))

    def _set_activation_function(self, activation):
        """Sets the final layer activation."""

        assert activation in AVAILABLE_OUTPUT_ACTIVATIONS, (
            "{} not among available output activation functions: "
            "{}".format(activation, ", ".join(
                AVAILABLE_OUTPUT_ACTIVATIONS)))

        self.activation = activation

        if activation == "sigmoid":
            self._activation = umath.sigmoid
        elif activation == "softmax":
            self._activation = umath.softmax
        else:
            raise KeyError("Final layer activation type '{}' not "
                           "recognized. Available activations:".format(
                               activation, ", ".join(
                                   AVAILABLE_OUTPUT_ACTIVATIONS)))

    @property
    def coef_(self):
        return self.coef

    @coef_.getter
    def coef_(self):
        return cp.deepcopy(self.coef)

    @coef_.setter
    def coef_(self, value):
        self.coef = value

    def fit(self, X_train, y_train, eta=1.0):
        """Performs a linear regression fit for data X_train and y_train.

        Args:
            X_train (ndarray): input data.
            y_train (ndarray): output one-hot labeled data.
            eta (float): learning rate, optional. Choices: float(constant), 
                "inverse". "Inverse" sets eta to 1 - i/(N+1). Default is 1.0.
        """
        X = cp.deepcopy(X_train)
        y = cp.deepcopy(y_train)

        self.N_features, self.p = X.shape
        _, self.N_labels = y.shape

        # Adds constant term and increments the number of predictors
        X = np.hstack([np.ones((self.N_features, 1)), X])
        self.p += 1

        # Adds beta_0 coefficients
        self.coef = np.zeros((self.p, self.N_labels))
        self.coef[0, :] = np.ones(self.N_labels)

        # Sets the learning rate
        self._set_learning_rate(eta)

        # Sets up method for storing cost function values
        self.cost_values = []
        self.cost_values.append(self._cost_function(X, y, self.coef))

        # Temp, clean this part up
        def learning_rate(t, t0, t1):
            return t0 / (t + t1)

        for i in range(self.max_iter):
            # Updates the learning rate
            eta_ = self._update_learning_rate(i, self.max_iter)

            # # OLD
            self.coef = self._gradient_descent(X, y, self.coef, eta_)
            # # self.coef += self._l2_regularization(self.coef)

            # Appends cost function values
            self.cost_values.append(self._cost_function(X, y, self.coef))

        self._fit_performed = True

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

        gradient = self._cost_function_gradient(X, y, weights) / X.shape[0]

        weights -= gradient*lr/X.shape[0]
        return weights

    def _cost_function(self, X, y, weights, eps=1e-15):
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

        p_probabilities = self._activation(y_pred)

        # Removes bad values and replaces them with limiting values eps
        p_probabilities = np.clip(p_probabilities, eps, 1-eps)

        # Sets up cross-entropy cost function for binary output
        cost1 = - y * np.log(p_probabilities)
        cost2 = (1 - y) * np.log(1 - p_probabilities)
        cost = np.sum(cost1 - cost2) + self._get_penalty(weights)*self.alpha

        return cost

    def _cost_function_gradient(self, X, y, weights):
        """Takes the gradient of the cost function w.r.t. the coefficients.

            dC(W)/dw = - X^T * (y - p(X^T * w))
        """
        grad = X.T @ (self._activation(self._predict(X, weights)) - y)
        grad += self.alpha*self._get_penalty_derivative(weights)
        return grad

    def _cost_function_laplacian(self, X, y, w):
        """Takes the laplacian of the cost function w.r.t. the coefficients.

            d^2C(w) / (w w^T) = X^T W X
        where
            W = p(1 - X^T * w) * p(X^T * w)
        """
        y_pred = self._predict(X, w)
        return X.T @ self._activation(1-y_pred) @ self._activation(y_pred) @ X

    def _predict(self, X, weights):
        """Performs a regular fit of parameters and sends them through 
        the sigmoid function.

        Args:
            X (ndarray): design matrix/feature matrix, shape (N, p)
            weights (ndarray): coefficients 
        """
        return X @ weights

    def score(self, X, y):
        """Returns the mean accuracy of the fit.

        Args:
            X (ndarray): array of shape (N, p - 1) to classify.
            Y (ndarray): true labels.

        Returns:
            (float): mean accuracy score for features_test values.
        """
        pred = self.predict(X)
        accuracies = np.sum(self._indicator(pred, y))

        return accuracies/float(y.shape[0])

    def _indicator(self, features_test, labels_test):
        """Returns 1 if features_test[i] == labels_test[i]

        Args:
            features_test (ndarray): array of shape (N, p - 1) to test for
            labels_test (ndarray): true labels

        Returns:
            (array): elements are 1 or 0
        """
        return np.where(features_test == labels_test, 1, 0)

    def predict(self, X):
        """Predicts category 1 or 2 of X.

        Args:
            X (ndarray): design matrix of shape (N, p - 1)
        """

        # predict(X)  Predict class labels for samples in X.

        if not self._fit_performed:
            raise UserWarning("Fit not performed.")

        # Adds intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Retrieves probabilitites
        probabilities = self._activation(self._predict(X, self.coef)).ravel()

        # Sets up binary probability
        results_proba = np.asarray([1 - probabilities, probabilities])

        # Moves axis from (2, N_probabilitites) to (N_probabilitites, 2)
        results_proba = np.moveaxis(results_proba, 0, 1)

        # Sets up binary prediction of either 0 or one
        results = np.where(results_proba[:, 0] >= results_proba[:, 1], 0, 1).T

        return results

    def predict_proba(self, X):
        """Predicts probability of a design matrix X of shape (N, p - 1)."""

        # predict_proba(X)    Probability estimates.

        if not self._fit_performed:
            raise UserWarning("Fit not performed.")

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        probabilities = self._activation(self._predict(X, self.coef)).ravel()
        results = np.asarray([1 - probabilities, probabilities])

        return np.moveaxis(results, 0, 1)


def __test_logistic_regression():
    from sklearn import datasets
    import sklearn.linear_model as sk_model
    import sklearn.model_selection as sk_modsel
    import matplotlib.pyplot as plt

    iris = datasets.load_iris()
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

    X_train, X_test, y_train, y_test = \
        sk_modsel.train_test_split(X, y, test_size=0.25, shuffle=True)

    # SK-Learn logistic regression
    sk_log_reg = sk_model.LogisticRegression(
        solver="liblinear", C=1.0, penalty="l2", max_iter=10000)
    sk_log_reg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
    X_new = np.linspace(0, 3, 100).reshape(-1, 1)
    y_sk_proba = sk_log_reg.predict_proba(X_new)

    print("SK-learn coefs: ", sk_log_reg.intercept_, sk_log_reg.coef_)

    # Manual logistic regression
    log_reg = LogisticRegression(penalty="l1", lr=1.0, max_iter=10000)
    log_reg.fit(cp.deepcopy(X_train), cp.deepcopy(
        y_train).reshape(-1, 1), eta="inverse")
    y_proba = log_reg.predict_proba(X_new)

    print("Manual coefs:", log_reg.coef_)

    # =========================================================================
    # Runs tests with SK learn's coefficients, and checks that our
    # implementation's predictions match SK-learn's predictions.
    # =========================================================================

    print("Score before using SK-learn's coefficients: {0:.16f}".format(
        log_reg.score(X_test, y_test)))

    # Sets the coefficients from the SK-Learn to local method
    log_reg.coef_ = np.asarray(
        [sk_log_reg.intercept_[0], sk_log_reg.coef_[0, 0]]).reshape((-1, 1))

    print("Score after using SK-learn's coefficients: {0:.16f}".format(
        log_reg.score(X_test, y_test)))

    # Asserts that predicted probabilities matches.
    y_sk_proba_compare = sk_log_reg.predict_proba(X_test)
    y_proba_compare = log_reg.predict_proba(X_test)
    assert np.allclose(y_sk_proba_compare, y_proba_compare), (
        "Predicted probabilities do not match: (SKLearn) {} != {} "
        "(local implementation)".format(y_sk_proba_compare, y_proba_compare))

    # Asserts that the labels match
    sk_predict = sk_log_reg.predict(X_test)
    local_predict = log_reg.predict(X_test)
    assert np.allclose(sk_predict, local_predict), (
        "Predicted class labels do not match: (SKLearn) {} != {} "
        "(local implementation)".format(sk_predict, local_predict))

    # Assert that the scores match
    sk_score = sk_log_reg.score(X_test, y_test)
    local_score = log_reg.score(X_test, y_test)
    assert np.allclose(sk_score, local_score), (
        "Predicted score do not match: (SKLearn) {} != {} "
        "(local implementation)".format(sk_score, local_score))

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
