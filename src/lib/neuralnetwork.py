import numpy as np
import copy as cp
import utils.math_tools as umath
from utils.math_tools import AVAILABLE_ACTIVATIONS, \
    AVAILABLE_OUTPUT_ACTIVATIONS, AVAILABLE_COST_FUNCTIONS


def plot_image(sample_, label, pred):
    """Simple function for plotting the input."""
    sample = cp.deepcopy(sample_)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    plt.imshow(
        sample.reshape(int(np.sqrt(sample.shape[0])),
                       int(np.sqrt(sample.shape[0]))),
        cmap=cm.gray)
    title_str = "Label: {} Prediction: {}".format(label, pred)
    print(title_str)
    plt.title(title_str)
    plt.show()


class MultilayerPerceptron:
    def __init__(self, layer_sizes, activation="sigmoid",
                 output_activation="sigmoid", cost_function="mse", alpha=0.0,
                 momentum=0.0):
        """Initializer for multilayer perceptron.

        Number of layers is always minimum N_layers + 2.

        Args:
            layer_sizes (list(int)): list of layer sizes after input data.
                Constists of [input_layer_size, N layer sizes, output_layer].
            activation (str): activation function. Choices is "sigmoid", 
                "identity", "relu", "tanh", "heaviside". Optional, default is 
                "sigmoid".
            output_activation (str): final layer activation function. Choices 
                is "sigmoid" or "sigmoid", "softmax", "identity". Optional, 
                default is "sigmoid".
            cost_function (str): Cost function. Choices is "mse", "log_loss". 
                Optional, default "mse".
            alpha (float): L2 regularization term. Default is 0.0.
            momentum (float): adds a dependency on previous gradient
        Raises:
            AssertionError: if input_data_size is not a list.
            AssertionError: if layer_sizes is less than two.
        """
        assert isinstance(layer_sizes, list), "must provide a layer size list"
        assert len(layer_sizes) >= 2, ("Must have at least two layers: "
                                       "len(layer_sizes)={}".format(
                                           len(layer_sizes)))

        self._set_layer_activation(activation)
        self._set_output_layer_activation(output_activation)
        self._set_cost_function(cost_function)

        # L2 regularization term
        self.alpha = alpha

        # Sets momentum given it is valid
        assert momentum >= 0.0, "momentum must be positive"
        self.momentum = momentum

        # Sets up weights and biases
        self.weights = [
            np.random.randn(l_j, l_i)
            for l_i, l_j in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(l_j, 1) for l_j in layer_sizes[1:]]

        self.layer_sizes = layer_sizes
        self.N_layers = len(layer_sizes)

    def _set_layer_activation(self, activation):
        """Sets the layer activation."""
        assert activation in AVAILABLE_ACTIVATIONS, (
            "{} not among available output activation functions: "
            "{}".format(activation, ", ".join(AVAILABLE_ACTIVATIONS)))

        self.activation = activation

        if activation == "sigmoid":
            self._activation = umath.sigmoid
            self._activation_derivative = umath.sigmoid_derivative
        elif activation == "identity":
            self._activation = umath.identity
            self._activation_derivative = umath.identity_derivative
        elif activation == "relu":
            self._activation = umath.relu
            self._activation_derivative = umath.relu_derivative
        elif activation == "tanh":
            self._activation = umath.tanh_
            self._activation_derivative = umath.tanh_derivative
        elif activation == "heaviside":
            self._activation = umath.heaviside
            self._activation_derivative = umath.heaviside_derivative
        else:
            raise KeyError("Activation type '{}' not recognized. Available "
                           "activations:".format(
                               activation, ", ".join(AVAILABLE_ACTIVATIONS)))

# TODO: final activation derivative never used when backpropagating? Derive the backpropegation again perhaps

    def _set_output_layer_activation(self, output_activation):
        """Sets the final layer activation."""

        assert output_activation in AVAILABLE_OUTPUT_ACTIVATIONS, (
            "{} not among available output activation functions: "
            "{}".format(output_activation, ", ".join(
                AVAILABLE_OUTPUT_ACTIVATIONS)))

        self.output_activation = output_activation

        if output_activation == "sigmoid":
            self._output_activation = umath.sigmoid
            self._output_activation_derivative = umath.sigmoid_derivative
        elif output_activation == "identity":
            self._output_activation = umath.identity
            self._output_activation_derivative = umath.identity_derivative
        elif output_activation == "softmax":
            self._output_activation = umath.softmax
            self._output_activation_derivative = umath.softmax_derivative
        else:
            raise KeyError("Final layer activation type '{}' not "
                           "recognized. Available activations:".format(
                               activation, ", ".join(
                                   AVAILABLE_OUTPUT_ACTIVATIONS)))

    def _set_learning_rate(self, eta):
        """Sets the learning rate."""
        if isinstance(eta, float):
            self._update_learning_rate = lambda _i, _N: eta
        elif eta == "inverse":
            self._update_learning_rate = lambda _i, _N: 1 - _i/float(_N+1)
        else:
            raise KeyError(("Eta {} is not recognized learning"
                            " rate.".format(eta)))

    def _set_cost_function(self, cost_function):
        """Sets the cost function to use.

        A nice list of different cost functions found here:
        https://stats.stackexchange.com/questions/154879/a-list-of-
        cost-functions-used-in-neural-networks-alongside-applications

        Args:
            cost_functions (str): name of the cost function to use.

        Raises:
            KeyError if cost_function is not a recognized cost function.
        """
        self.cost_function = cost_function
        if cost_function == "mse":
            self._base_cost_function = umath.mse_cost
            self._base_cost_function_derivative = umath.mse_cost_derivative
        elif cost_function == "log_loss":
            self._base_cost_function = umath.log_entropy
            self._base_cost_function_derivative = umath.log_entropy_derivative
        elif cost_function == "exponential_cost":
            self._base_cost_function = umath.exponential_cost
            self._base_cost_function_derivative = \
                umath.exponential_cost_derivative
        elif cost_function == "hellinger_distance":
            raise NotImplementedError(cost_function)
        elif cost_function == "kullback_leibler_divergence":
            raise NotImplementedError(cost_function)
        elif cost_function == "generalized_kullback_leibler_divergence":
            raise NotImplementedError(cost_function)
        elif cost_function == "itakura_saito_distance":
            raise NotImplementedError(cost_function)
        else:
            raise KeyError("Cost function '{}' not recognized. Available loss"
                           " functions: {}".format(cost_function, ", ".join(
                               AVAILABLE_COST_FUNCTIONS)))

    def _cost_function(self, layer, x, y):
        """Cost function.

        Args:
            layer (int): which layer we are in
            x (ndarray): input values.
            y (ndarray): output values, one-hot vector.
        """
        base_cost = self._base_cost_function(x, y)

        return base_cost + self._regularization(layer)*0.5/x.shape[0]

    def _cost_function_derivative(self, y, y_true):
        """Derivative of the cost function.

        Args:
            y (ndarray): input values.
            y_true (ndarray): output values, one-hot vector.
        """
        base_cost_derivative = self._base_cost_function_derivative(y, y_true)

        return base_cost_derivative

    def _regularization(self, layer):
        """Computes the L2 regularization.

        Args:
            layer (int): layer to compute regularization for.

        Returns:
            (float) l2-norm of given layer.
        """
        if self.alpha != 0.0:
            return self.alpha*np.sum(self.weights[layer]**2)
        else:
            return 0.0

    def _regularization_derivative(self, layer):
        """Computes the L2 regularization derivative.

        Args:
            layer (int): layer to compute regularization for.

        Returns:
            (ndarray) derivative of the l2-norm of given layer.
        """
        if self.alpha != 0.0:
            return self.alpha*self.weights[layer]
        else:
            return 0.0

    def predict(self, x):
        """Returns the last layer of activation from _forward_pass."""
        return self._forward_pass(x)[-1]

    def _forward_pass(self, activation):
        """Performs a feed-forward to the last layer."""
        activations = [activation]
        for i in range(self.N_layers - 1):
            z = (self.weights[i] @ activations[i])
            z += self.biases[i]

            if i+1 != (self.N_layers - 1):
                activations.append(self._activation(z))

        activations.append(self._output_activation(z))

        return activations

    def _back_propagate(self, x, y):
        """Performs back-propagation on a single dataset.

        Args:
            x (ndarray): initial layer input.
            y (ndarray): true output values(labels), one-hot vector.

        Returns:
            (list(ndarray)): all layer weight gradients
            (list(ndarray)): all layer bias gradients
        """

        # Retrieves the z and sigmoid for each layer in sample
        z_list = []
        self.activations = [x]
        for i in range(self.N_layers - 1):
            z = self.weights[i] @ self.activations[i]
            z += self.biases[i]
            z_list.append(z)

            if (i+1) != (self.N_layers - 1):
                # Middle layer(s) activation
                self.activations.append(self._activation(z))
            else:
                # Sigmoid output layer
                self.activations.append(self._output_activation(z).T)

        # Backpropegation begins, initializes the backpropagation derivatives
        delta_w = [np.empty(w.shape) for w in self.weights]
        delta_b = [np.empty(b.shape) for b in self.biases]

        # Gets initial delta value, first of the four equations
        delta = self._cost_function_derivative(self.activations[-1], y).T

        # No final derivative?
        # delta *= self._output_activation_derivative(z_list[-1])

        # Sets last element before back-propagating
        delta_b[-1] = delta
        delta_w[-1] = delta @ self.activations[-2].T
        delta_w[-1] += self._regularization_derivative(-1)  # /x.shape[0]

        # delta_b, delta_w = self._compute_gradient_derivatives(layer, delta_b,
        #     delta_w, delta)

        # Loops over layers
        for l in range(2, self.N_layers):
            # Second equation: delta^l = delta^{l+1} W^l * dsigma(z^l)
            # Retrieves the z and gets it's derivative
            z = z_list[-l]
            sp = self._activation_derivative(z)

            # Sets up delta^l
            delta = self.weights[-l+1].T @ delta
            delta *= sp

            delta_b[-l] = delta  # np.sum(delta, axis=1)
            delta_w[-l] = delta @ self.activations[-l-1].T
            delta_w[-l] += self._regularization_derivative(-l)

        return delta_w, delta_b

    def train(self, data_train, data_train_labels, epochs=10,
              mini_batch_size=50, eta=1.0, data_test=None,
              data_test_labels=None):
        """Trains the neural-net on provided data. Assumes data size 
        is the same as what provided in the initialization.

        Uses Stochastic Gradient Descent(SGA) and mini-batches to get the 
        deed done.

        Args:
            data_train (ndarray): training data. Shape: 
                (samples, input_size, 1)
            data_train_labels (ndarray): training data labels. Shape: 
                (samples, output_size)
            epochs (int): number of times we are to train the data. Default 
                is 10.
            mini_batch_size (int): size of mini batch. Optional, default is 50.
            eta (float): learning rate, optional. Choices: float(constant), 
                "inverse". "Inverse" sets eta to 1 - i/(N+1). Default is 1.0.
            data_test (ndarray): data to run tests for. Shape:
                (samples, input_size, 1)
            data_test_labels (ndarray): training data labels. Shape: 
                (samples, output_size)

        Raises:
            AssertionError: if input data to not match the specified layer 
                data given in the initialization.
        """

        assert self.layer_sizes[0] == data_train.shape[1], (
            "training data "
            "and labels do not match in shape: {} != {}".format(
                self.layer_sizes[0], data_train.shape[1]))

        # Sets if we are to evaluate the data while running
        if (not isinstance(data_test, type(None))) and \
                (not isinstance(data_test_labels, type(None))):
            perform_eval = True
        else:
            perform_eval = False

        N_train_size = data_train.shape[0]

        # Gets the number of batches
        number_batches = N_train_size // mini_batch_size

        self._set_learning_rate(eta)

        for epoch in range(epochs):

            # Updates the learning rate
            eta_ = self._update_learning_rate(epoch, epochs)

            # Performs the SGA step of shuffling data
            shuffle_indexes = np.random.choice(list(range(N_train_size)),
                                               size=N_train_size,
                                               replace=False)

            # Shuffles the data with the shuffle-indices
            shuffled_data = data_train[shuffle_indexes]
            shuffled_labels = data_train_labels[shuffle_indexes]

            # Splits data into minibatches
            shuffled_data = [
                shuffled_data[i:i+mini_batch_size]
                for i in range(0, N_train_size, number_batches)]
            shuffled_labels = [
                shuffled_labels[i:i+mini_batch_size]
                for i in range(0, N_train_size, number_batches)]

            # Iterates over mini batches
            for mb_data, mb_labels in zip(shuffled_data, shuffled_labels):

                self.update_mini_batch(mb_data, mb_labels, eta_)

            # If we have provided testing data, we perform an epoch evaluation
            if perform_eval:
                print("Epoch: {} Score: {}/{}".format(
                    epoch, self.evaluate(data_test, data_test_labels),
                    len(data_test_labels)))
            else:
                print("Epoch {}".format(epoch))

    def update_mini_batch(self, mb_data, mb_labels, eta):
        """Trains the network on the mini batch."""

        # Resets gradient sums
        delta_w_sum = [np.zeros(w.shape) for w in self.weights]
        delta_b_sum = [np.zeros(b.shape) for b in self.biases]

        # Loops over all samples and labels in mini batch
        for sample, label in zip(mb_data, mb_labels):

            # Runs back-propagation
            delta_w, delta_b = self._back_propagate(sample, label)

            # Sums the derivatives into a single list of derivative-arrays.
            delta_w_sum = [dw + dws for dw, dws in zip(delta_w, delta_w_sum)]
            delta_b_sum = [db + dbs for db, dbs in zip(delta_b, delta_b_sum)]

        # Updates weights and biases by subtracting their gradients
        for l in range(self.N_layers - 1):
            self.weights[l] -= (delta_w_sum[l]*eta/len(mb_data))
            self.biases[l] -= (delta_b_sum[l]*eta/len(mb_data))

    def evaluate(self, test_data, test_labels, show_image=False):
        """Evaluates test data.

        Args:
            test_data (ndarray): array of shape (sample, input_size, 1), 
                contains the input data to test for.
            test_labels (ndarray): array of desired output to compare against.
                On the shape of (sample, output_size)
            show_image (bool): plots input values. Assumes input is square. 
                Optional, default is False.
        """

        results = []
        for test, label in zip(test_data, test_labels):
            pred = self.predict(np.atleast_2d(test))
            results.append(int(np.argmax(pred) == np.argmax(label)))

            if show_image:
                plot_image(test, np.argmax(label), np.argmax(pred))

        return sum(results)

    def score(self, test_data, test_labels, verbose=False):
        """Returns the accuracy score for given test data.

        Args:
            test_data (ndarray): array of shape (sample, input_size, 1), 
                contains the input data to test for.
            test_labels (ndarray): array of desired output to compare against.
                On the shape of (sample, output_size)        
        """
        results = self.evaluate(test_data, test_labels)
        if verbose:
            print("Accuracy = {}/{} = {}".format(results, len(results),
                                                 results/len(results)))
        return results/len(results)


def __test_mlp_mnist():
    import gzip
    import pickle

    test_data_path = "../../datafiles/HandwritingClassification/mnist.pkl"
    with open(test_data_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data_train, data_valid, data_test = u.load()

    print("DATA TRAIN: ", data_train[0].shape, data_train[1].shape)
    print("DATA VALID: ", data_valid[0].shape, data_valid[1].shape)
    print("DATA TEST: ", data_test[0].shape, data_test[1].shape)

    def convert_output(label_, output_size):
        """Converts label to output vector."""
        y_ = np.zeros(output_size, dtype=float)
        y_[label_] = 1.0
        return y_

    # Converts data to ((N, p-1)) shape.
    data_train_samples = np.asarray(
        [d_.reshape((-1, 1)) for d_ in data_train[0]])
    data_valid_samples = np.asarray(
        [d_.reshape((-1, 1)) for d_ in data_valid[0]])
    data_test_samples = np.asarray(
        [d_.reshape((-1, 1)) for d_ in data_test[0]])

    # Converts labels from single floats to arrays with 1.0 at correct output.
    # Aka, to one-hot vector format.
    data_train_labels = np.asarray(
        [convert_output(l, 10) for l in data_train[1]])
    data_valid_labels = np.asarray(
        [convert_output(l, 10) for l in data_valid[1]])
    data_test_labels = np.asarray(
        [convert_output(l, 10) for l in data_test[1]])

    # Sets up my MLP.
    MLP = MultilayerPerceptron([data_train_samples.shape[1], 50, 10],
                               activation="sigmoid",
                               cost_function="log_loss",
                               output_activation="sigmoid",
                               alpha=0.0)
    MLP.train(data_train_samples, data_train_labels,
              data_test=data_test_samples,
              data_test_labels=data_test_labels,
              mini_batch_size=20,
              epochs=100,
              eta="inverse")
    MLP.evaluate(data_test_samples, data_test_labels, show_image=True)


def __test_nn_sklearn_comparison():
    import warnings
    import copy as cp
    from sklearn.neural_network import MLPRegressor

    def test_regressor(X_train, y_train, X_test, y_test, nn_layers,
                       sk_hidden_layers, input_activation, output_activation,
                       alpha=0.0):

        if input_activation == "sigmoid":
            sk_input_activation = "logistic"
        else:
            sk_input_activation = input_activation

        if output_activation == "sigmoid":
            sk_output_activation = "logistic"
        else:
            sk_output_activation = output_activation

        mlp = MLPRegressor(
            solver='sgd',               # Stochastic gradient descent.
            activation=sk_input_activation,  # Skl name for sigmoid.
            alpha=alpha,                  # No regularization for simplicity.
            hidden_layer_sizes=sk_hidden_layers)  # Full NN size is (1,3,3,1).

        mlp.out_activation_ = sk_output_activation

        # Force sklearn to set up all the necessary matrices by fitting a data
        # set. We dont care if it converges or not, so lets ignore raised
        # warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_train, y_train)

        # =====================================================================
        n_samples, n_features = X_train.shape
        batch_size = n_samples
        hidden_layer_sizes = mlp.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        layer_units = ([n_features] + hidden_layer_sizes + [mlp.n_outputs_])
        activations = [X_test]
        activations.extend(np.empty((batch_size, n_fan_out))
                           for n_fan_out in layer_units[1:])
        deltas = [np.empty_like(a_layer) for a_layer in activations]
        coef_grads = [np.empty((n_fan_in_, n_fan_out_))
                      for n_fan_in_, n_fan_out_ in zip(layer_units[:-1],
                                                       layer_units[1:])]
        intercept_grads = [np.empty(n_fan_out_)
                           for n_fan_out_ in layer_units[1:]]
        # =====================================================================

        mlp.out_activation_ = sk_output_activation
        activations = mlp._forward_pass(activations)
        loss, coef_grads, intercept_grads = mlp._backprop(
            X_test, y_test, activations, deltas, coef_grads, intercept_grads)

        # Activates my own MLP
        nn = MultilayerPerceptron(
            nn_layers, activation=input_activation,
            output_activation=output_activation, alpha=alpha)

        # Copy the weights and biases from the scikit-learn network to your
        # own.
        for i, w in enumerate(mlp.coefs_):
            nn.weights[i] = cp.deepcopy(w.T)
        for i, b in enumerate(mlp.intercepts_):
            nn.biases[i] = cp.deepcopy(b.T.reshape(-1, 1))

        # Call your own backpropagation function, and you're ready to compare
        # with the scikit-learn code.
        y_sklearn = mlp.predict(X_test)
        y = nn.predict(cp.deepcopy(X_test).T)

        # Asserts that the forward pass is correct
        assert np.allclose(y, y_sklearn), (
            "Prediction {} != {}".format(y, y_sklearn))

        delta_w, delta_b = nn._back_propagate(X_test.T, y_test)

        # Asserts that the the activations is correct in back propagation
        for i, a in enumerate(nn.activations):
            print(i, a.T, activations[i])
            assert np.allclose(
                a.T, activations[i]), "error in layer {}".format(i)
        else:
            print("Activations are correct.")

        # Asserts that the the biases is correct in back propagation
        for i, derivative_bias in enumerate(delta_b):
            print(i, derivative_bias.T, intercept_grads[i])
            assert np.allclose(
                derivative_bias.T, intercept_grads[i]), (
                "error in layer {}".format(i))
        else:
            print("Biases derivatives are correct.")

        # Asserts that the the weights is correct in back propagation
        for i, derivative_weight in enumerate(delta_w):
            print(i, derivative_weight.T, coef_grads[i])
            assert np.allclose(derivative_weight.T,
                               coef_grads[i]), "error in layer {}".format(i)
        else:
            print("Weight derivatives are correct.")

        print("Test complete\n")

    # Training data
    X_train1 = np.array([[0.0], [1.0]])
    y_train1 = np.array([0, 2])
    layer_sizes1 = [1, 3, 3, 1]
    sk_hidden_layers1 = (3, 3)

    X_train2 = np.array([[0.0, 0.5], [1.0, 1.5]])
    y_train2 = np.array([0, 1.0])
    layer_sizes2 = [2, 3, 3, 2]
    sk_hidden_layers2 = (3, 3)

    X_train3 = np.random.rand(100, 10)
    y_train3 = np.random.rand(100)
    layer_sizes3 = [10, 20, 20, 10]
    sk_hidden_layers3 = (20, 20)

    # Completely random data point(s) which we will propagate through
    # the network.
    X_test1 = np.array([[1.125982598]])
    y_test1 = np.array([8.29289285])

    X_test2 = np.array([[1.125982598, 2.937172838]])
    y_test2 = np.array([8.29289285])

    X_test3 = np.array([np.random.rand(10)])
    y_test3 = np.array([8.29289285])

    test_regressor(X_train1, y_train1, X_test1, y_test1,
                   layer_sizes1, sk_hidden_layers1, "sigmoid", "softmax")
    test_regressor(X_train2, y_train2, X_test2, y_test2,
                   layer_sizes2, sk_hidden_layers2, "sigmoid", "identity",
                   alpha=0.5)
    test_regressor(X_train3, y_train3, X_test3, y_test3,
                   layer_sizes3, sk_hidden_layers3, "sigmoid", "sigmoid")

    print("Forward and back propagation tests passed.\n")


if __name__ == '__main__':
    # __test_mlp_mnist()
    __test_nn_sklearn_comparison()
