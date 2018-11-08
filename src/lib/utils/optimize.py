#!/usr/bin/env python3

import numpy as np
import abc
import warnings
from scipy.optimize import minimize
from scipy.special import expit

OPTIMIZERS = ["GradientDescent", "LogRegGradientDescent",
              "ConjugateGradient", "SGA", "NewtonRaphson"]

OPTIMIZERS_KEYWORDS = ["lr-gd", "gd", "cg", "sga", "sga-mb", "nr"]


class _OptimizerBase(abc.ABC):
    """Base class for optimization."""

    def __init__(self, momentum=0.0):
        """Basic initialization."""
        self.momentum = momentum
        if momentum != 0.0:
            warnings.warn("momentum", NotImplementedError)

    def _set_learning_rate(self, eta):
        """Sets the learning rate."""
        if isinstance(eta, float):
            self._update_learning_rate = lambda _i, _N: eta
        elif eta == "inverse":
            self._update_learning_rate = lambda _i, _N: 1 - _i/float(_N+1)
        # elif eta == "scaling":
        #     self._update_learning_rate =
        else:
            raise KeyError(("Eta {} is not recognized learning"
                            " rate.".format(eta)))

    # Abstract class methods makes it so that they MUST be overwritten by child
    @abc.abstractmethod
    def solve(self, X, y, coef, cf, cf_prime, eta=1.0, max_iter=10000,
              store_coefs=False, tol=1e-15):
        """General solve method.

        Args:
            X (ndarray): design matrix, shape (N_inputs, p-1).
            y (float): true output.
            coef (ndarray): beta coefficients, shape (p, classes/labels)
            cf (func): cost function.
            cf_prime (func): cost function gradient.
            eta (float/str): learning rate, optional. Choices: constant float
                or 'inverse'. Default is 1.0.
            max_iter (int): maximum number of allowed iterations, optional.
                Default is 10000.
            store_coefs (bool): store the coefficients as they are calculated.
                Default is False.
        """

        # Sets the learning rate
        self._set_learning_rate(eta)

        if store_coefs:
            self.coefs = np.zeros((max_iter, *coef.shape))

        # Sets up method for storing cost function values
        self.cost_values = np.empty(max_iter+1)

        # Initial guess
        self.cost_values[0] = cf(X, y, coef)


class LogRegGradientDescent(_OptimizerBase):
    """Class tailored to logistic regression."""

    def solve(self, X, y, coef, cf, cf_prime, eta=1e-4, max_iter=1000,
              store_coefs=False, tol=1e-4, alpha=1.0, scale=1.0):
        """Gradient descent solver.
        """
        super().solve(X, y, coef, cf, cf_prime, eta, max_iter, store_coefs)
        # Initialize beta-parameters
        beta = np.zeros(X.shape[1])
        beta_prev = beta.copy()

        norm = 100
        eta_k = 0

        # first step
        z = np.dot(X, beta)
        p = expit(z)
        gradient = -np.dot(X.T, y-p)/scale + alpha*beta/scale
        beta -= eta*gradient
        norm = np.linalg.norm(gradient)

        gradient_prev = gradient.copy()

        for k in range(1, max_iter):

            z = np.dot(X, beta)
            p = expit(z)

            gradient_prev = gradient.copy()
            gradient = -np.dot(X.T, y-p)/scale + alpha*beta/scale

            eta_k = np.dot((beta - beta_prev), gradient-gradient_prev) / \
                np.linalg.norm(gradient-gradient_prev)**2

            beta_prev = beta.copy()
            beta -= eta_k*gradient

            norm = np.linalg.norm(gradient)

            # if(k % 10 == 0):
            #     print(norm, scale*norm)

            if(scale*norm < tol):
                return beta
        else:
            warnings.warn(("Solution did not converge for i={}"
                           " iterations".format(max_iter)), RuntimeWarning)
            return beta

        # print (eta)
        # coef_prev = np.zeros(coef.shape)

        # for i in range(max_iter):

        #     if np.abs(np.sum(coef - coef_prev)) < tol:
        #         print("exits: i=", i, "coef:", coef, " diff:",
        #               np.abs(np.sum(coef - coef_prev)))
        #         return coef

        #     coef_prev = coef

        #     # Updates the learning rate
        #     eta_ = self._update_learning_rate(i, max_iter)

        #     # Updates coeficients using a gradient descent step
        #     coef = self._gradient_descent_step(X, y, coef, cf_prime, eta_)

        #     # Adds cost function value
        #     self.cost_values[i] = cf(X, y, coef)

        #     if store_coefs:
        #         self.coefs[i] = coef
        # else:
        #     warnings.warn(("Solution did not converge for i={}"
        #                    " iterations".format(max_iter)), RuntimeWarning)
        #     return coef

    @staticmethod
    def _gradient_descent_step(X, y, coef, cf_prime, eta):
        """Performs a single gradient descent step."""
        gradient = cf_prime(X, y, coef)
        coef = coef - gradient*eta  # / X.shape[0]
        return coef


class GradientDescent(_OptimizerBase):
    """Class tailored to logistic regression."""

    def solve(self, X, y, coef, cf, cf_prime, eta=1.0, max_iter=1000,
              store_coefs=False, tol=1e-15, alpha=1.0):
        """Gradient descent solver.
        """
        super().solve(X, y, coef, cf, cf_prime, eta, max_iter, store_coefs)

        coef_prev = coef.copy()

        for i in range(max_iter):

            coef_prev = coef

            # Updates the learning rate
            eta_ = self._update_learning_rate(i, max_iter)

            # Updates coeficients using a gradient descent step
            coef = self._gradient_descent_step(X, y, coef, cf_prime, eta_)

            # Adds cost function value
            self.cost_values[i] = cf(X, y, coef)

            if store_coefs:
                self.coefs[i] = coef

            if np.abs(np.sum(coef - coef_prev)) < tol:
                # print("exits: i=", i, "coef:", coef, " diff:",
                #       np.abs(np.sum(coef - coef_prev)))
                return coef

        else:
            warnings.warn(("Solution did not converge for i={}"
                           " iterations".format(max_iter)), RuntimeWarning)
            return coef

    @staticmethod
    def _gradient_descent_step(X, y, coef, cf_prime, eta):
        """Performs a single gradient descent step."""
        gradient = cf_prime(X, y, coef)
        coef = coef - gradient*eta  # / X.shape[0]
        return coef


class ConjugateGradient(_OptimizerBase):
    """Conjugate gradient solver."""

    def solve(self, X, y, coef, cf, cf_prime, eta=1.0, max_iter=1000,
              store_coefs=False, tol=1e-15, alpha=1.0):
        super().solve(X, y, coef, cf, cf_prime, eta, max_iter, store_coefs)

        def f(coef_, X_, y_): cf(X_, y_, coef_)

        def f_prime(coef_, X_, y_): cf_prime(X_, y_, coef_)
        opt = minimize(f, coef, args=(X, y), jac=f_prime, method="CG",
            tol=tol, options={"maxiter": max_iter})
        return opt.x


class SGA(_OptimizerBase):
    """Stochastic gradient descent solver."""

    def __init__(self, use_minibatches=False, mini_batch_size=50, **kwargs):
        super().__init__(**kwargs)
        self.use_minibatches = use_minibatches
        self.mini_batch_size = mini_batch_size

    def solve(self, X, y, coef, cf, cf_prime, eta=1.0, max_iter=1000,
              store_coefs=False, tol=1e-15, alpha=1.0):

        super().solve(X, y, coef, cf, cf_prime, eta, max_iter, store_coefs)

        N_size = X.shape[0]

        if self.use_minibatches:
            number_batches = N_size // self.mini_batch_size

        for i in range(max_iter):

            # Updates the learning rate
            eta_ = self._update_learning_rate(i, max_iter)

            # Performs the SGA step of shuffling data
            shuffle_indexes = np.random.choice(list(range(N_size)),
                                               size=N_size,
                                               replace=False)

            # Shuffles the data with the shuffle-indices
            shuffled_X = X[shuffle_indexes]
            shuffled_y = y[shuffle_indexes]

            if self.use_minibatches:
                # Splits data into minibatches
                shuffled_X = [
                    shuffled_X[i:i+self.mini_batch_size, :]
                    for i in range(0, N_size, number_batches)]
                shuffled_y = [
                    shuffled_y[i:i+self.mini_batch_size, :]
                    for i in range(0, N_size, number_batches)]

                for mb_X, mb_y in zip(shuffled_X, shuffled_y):
                    # coef = self._update_coef(mb_X, mb_y, coef, cf_prime, eta_)
                    coef = GradientDescent._gradient_descent_step(
                        mb_X, mb_y, coef, cf_prime, eta_)

            else:
                # coef = self._update_coef(shuffled_X, shuffled_y, coef,
                #                          cf_prime, eta_)
                coef = GradientDescent._gradient_descent_step(
                    shuffled_X, shuffled_y, coef, cf_prime, eta_)

            # Adds cost function value
            self.cost_values[i] = cf(X, y, coef)

            if store_coefs:
                self.coefs[i] = coef

        return coef


class NewtonRaphson(_OptimizerBase):
    """Newton-Raphson solver."""

    def solve(self, X, y, coef, cf, cf_prime, eta=1.0, max_iter=10000,
              store_coefs=False, tol=1e-15, alpha=1.0):

        super().solve(X, y, coef, cf, cf_prime, eta, max_iter, store_coefs)

        def f(coef_, X_, y_): 
            return cf(X_, y_, coef_)

        def f_prime(coef_, X_, y_): 
            return cf_prime(X_, y_, coef_)
        opt = minimize(f, coef, args=(X, y), jac=f_prime, method="Newton-CG",
            tol=tol, options={"maxiter": max_iter})
        return opt.x
        # eta = 1e-3
        # coef_prev = np.zeros(coef.shape)

        # for i in range(max_iter):

        #     eta_ = self._update_learning_rate(i, max_iter)

        #     f = cf(X, y, coef)
        #     f_prime = cf_prime(X, y, coef)

        #     if np.linalg.norm(f_prime) < 1e-14:
        #         raise RuntimeError("Divide by zero.")

        #     dx = f / f_prime

        #     # print (coef, dx, cf(X, y, coef), cf_prime(X, y, coef) )

        #     coef_prev = coef - dx*eta

        #     if i % 100 == 0:
        #         print(i, coef, dx, f, f_prime)
        #         # print(np.linalg.norm(coef - coef_prev)**2)

        #     # Checks if we have convergence
        #     if np.linalg.norm(dx) < tol:
        #         print("exits at i={} with dx={}. Diff={}".format(
        #             i, dx, np.abs(coef_prev - coef).sum()))
        #         return coef

        #     coef = coef_prev

        # else:
        #     # If no convergence is reached, raise a warning.
        #     warnings.warn("Solution did not converge", RuntimeWarning)
        #     return coef


def _test_minimizers():
    import copy as cp

    max_iter=int(1e5)
    tol = 1e-8
    eta = 1e-2

    def f(_a, _b, x):
        # return x**2 - 612
        return x**4 - 3*x**3 + 2

    def f_prime(_a, _b, x):
        # return 2*x
        return 4*x**3 - 9*x**2

    answer = 2.25

    x = np.array([2.323])
    a = np.array([1.0])
    b = np.array([3.0])

    SDSolver = GradientDescent()
    SD_x0 = SDSolver.solve(cp.deepcopy(x), a, b, f,
                           f_prime, eta=eta, tol=tol, max_iter=max_iter)

    NR_Solver = NewtonRaphson()
    NR_x0 = NR_Solver.solve(cp.deepcopy(x), a, b, f,
                            f_prime, eta=eta, tol=tol, max_iter=max_iter)

    # SGA_Solver = SGA()
    # SGA_x0 = SGA_Solver.solve(x, a, b, f, f_prime, eta=1e-2, max_iter=int(1e6))

    # SGA_MB_Solver = SGA(mini_batch_size=True)
    # SGA_MB_x0 = SGA_MB_Solver.solve(x, a, b, f, f_prime, eta=1e-2, max_iter=int(1e6))

    print("GradientDescent", f(a, b, SD_x0), f_prime(a, b, SD_x0), SD_x0)
    print("Newton-CG", f(a,b,NR_x0), f_prime(a,b,NR_x0), NR_x0)

    assert np.abs(SD_x0[0] - answer) < 1e-5, (
        "GradientDescent is incorrect: {}".format(SD_x0[0]))
    # assert np.abs(SGA_x0 - answer) < 1e-10, "SGA is incorrect"
    # assert np.abs(SGA_MB_x0 - answer) < 1e-10, "SGA MB is incorrect"
    assert np.abs(NR_x0[0] - answer) < 1e-5, (
        "Newton-Raphson is incorrect: {}".format(NR_x0[0]))

    print("All methods converged.")


if __name__ == '__main__':
    _test_minimizers()