#!/usr/bin/env python3

import numpy as np
import abc
import warnings

OPTIMIZERS = ["GradientDescent",
              "ConjugateGradient", "SGA", "NewtonRaphson"]

OPTIMIZERS_KEYWORDS = ["gd", "cg", "sga", "sga-mb", "nr"]


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


class GradientDescent(_OptimizerBase):
    def solve(self, X, y, coef, cf, cf_prime, eta=1.0, max_iter=1000,
              store_coefs=False, tol=1e-15):
        """Gradient descent solver.
        """
        super().solve(X, y, coef, cf, cf_prime, eta, max_iter, store_coefs)

        coef_prev = np.zeros(coef.shape)

        for i in range(max_iter):

            if np.abs(np.sum(coef - coef_prev)) < tol:
                # print("exits: i=", i, "coef:", coef, " diff:",
                #       np.abs(np.sum(coef - coef_prev)))
                return coef

            coef_prev = coef

            # Updates the learning rate
            eta_ = self._update_learning_rate(i, max_iter)

            # Updates coeficients using a gradient descent step
            coef = self._gradient_descent_step(X, y, coef, cf_prime, eta_)

            # Adds cost function value
            self.cost_values[i] = cf(X, y, coef)

            if store_coefs:
                self.coefs[i] = coef

        return coef

    @staticmethod
    def _gradient_descent_step(X, y, coef, cf_prime, eta):
        """Performs a single gradient descent step."""
        gradient = cf_prime(X, y, coef)
        coef = coef - gradient*eta / X.shape[0]
        return coef


class ConjugateGradient(_OptimizerBase):
    """Conjugate gradient solver."""

    def solve(self, X, y, coef, cf, cf_prime, eta=1.0, max_iter=1000,
              store_coefs=False, tol=1e-15):
        raise NotImplementedError("NewtonRaphson")
        super().solve(X, y, coef, cf, cf_prime, eta, max_iter, store_coefs)
        return scipy.optimize(f, x0, method="CG")


class SGA(_OptimizerBase):
    """Stochastic gradient descent solver."""

    def __init__(self, use_minibatches=False, mini_batch_size=50, **kwargs):
        super().__init__(**kwargs)
        self.use_minibatches = use_minibatches
        self.mini_batch_size = mini_batch_size

    def solve(self, X, y, coef, cf, cf_prime, eta=1.0, max_iter=1000,
              store_coefs=False, tol=1e-15):

        super().solve(X, y, coef, cf, cf_prime, eta, max_iter, store_coefs)

        N_size = X.shape[0]

        if self.use_minibatches:
            number_batches = N_size // self.mini_batch_size

        for i in range(max_iter):
            if i % 100 == 0:
                print(i)
            # Updates the learning rate
            eta_ = self._update_learning_rate(i, max_iter)

            # Performs the SGA step of shuffling data
            shuffle_indexes = np.random.choice(list(range(N_size)),
                                               size=N_size,
                                               replace=False)

            # Shuffles the data with the shuffle-indices
            shuffled_X = X[shuffle_indexes, :]
            shuffled_y = y[shuffle_indexes, :]

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
              store_coefs=False, tol=1e-14):

        super().solve(X, y, coef, cf, cf_prime, eta, max_iter, store_coefs)

        coef_prev = np.zeros(coef.shape)

        for i in range(max_iter):

            eta_ = self._update_learning_rate(i, max_iter)

            f = cf(X, y, coef)
            f_prime = cf_prime(X, y, coef)

            if abs(f_prime) < 1e-14:
                raise RuntimeError("Divide by zero.")

            dx = f / f_prime

            # print (coef, dx, cf(X, y, coef), cf_prime(X, y, coef) )

            coef_prev = coef - dx*eta

            if i % 100 == 0:
                print(i, coef, dx, f, f_prime)
                # print(np.linalg.norm(coef - coef_prev)**2)
            print (coef_prev, coef, dx)

            # Checks if we have convergence
            if np.abs(coef_prev - coef).sum() < tol:
                print("exits at i={} with dx={}. Diff={}".format(i, dx, np.abs(coef_prev - coef).sum()))
                return coef

            coef = coef_prev

        else:
            # If no convergence is reached, raise a warning.
            warnings.warn("Solution did not converge", RuntimeWarning)
            return coef


def _test_minimizers():
    import copy as cp

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
    SD_x0 = SDSolver.solve(cp.deepcopy(x), a, b, f, f_prime, eta=1e-2, max_iter=int(1e6))

    NR_Solver = NewtonRaphson()
    NR_x0 = NR_Solver.solve(cp.deepcopy(x), a, b, f, f_prime, eta=1e-2, max_iter=int(1e6))

    # SGA_Solver = SGA()
    # SGA_x0 = SGA_Solver.solve(x, a, b, f, f_prime, eta=1e-2, max_iter=int(1e6))

    # SGA_MB_Solver = SGA(mini_batch_size=True)
    # SGA_MB_x0 = SGA_MB_Solver.solve(x, a, b, f, f_prime, eta=1e-2, max_iter=int(1e6))

    print(f(a,b,SD_x0), f_prime(a,b,SD_x0))
    print(f(a,b,NR_x0), f_prime(a,b,NR_x0))

    assert np.abs(SD_x0[0] - answer) < 1e-10, (
        "GradientDescent is incorrect: {}".format(SD_x0[0]))
    # assert np.abs(SGA_x0 - answer) < 1e-10, "SGA is incorrect"
    # assert np.abs(SGA_MB_x0 - answer) < 1e-10, "SGA MB is incorrect"
    assert np.abs(NR_x0[0] - answer) < 1e-10, (
        "Newton-Raphson is incorrect: {}".format(NR_x0[0]))


if __name__ == '__main__':
    _test_minimizers()
