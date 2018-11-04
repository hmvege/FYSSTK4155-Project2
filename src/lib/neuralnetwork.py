import numpy as np
import copy as cp
from utils.math import sigmoid, sigmoid_derivative, mse_cost, \
    mse_cost_derivative, identity, identity_derivative
from tqdm import tqdm

# TODO: Implement a multilayer perceptron neural network here!


# def sigmoid(z):
#     return 1.0/(1.0 + np.exp(-z))


# def sigmoid_derivative(z):
#     s = sigmoid(z)
#     return s*(1-s)
#     # exp_ = np.exp(-z)
#     # return exp_ / (1 + exp_)**2


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
                 final_activation="identity", cost_function="mse", alpha=0.0):
        """Initializer for multilayer perceptron.

        Number of layers is always minimum N_layers + 2.

        Args:
            layer_sizes (list(int)): list of layer sizes after input data.
                Constists of [input_layer_size, N layer sizes, output_layer].
            activation (str): activation function. Choices is "sigmoid", 
                "identity", "relu", "tanh", "heaviside". Optional, default is 
                "sigmoid".
            final_activation (str): final layer activation function. Choices 
                is "sigmoid" or "logistic", "softmax", "identity", "relu", "tanh", 
                "heaviside". Optional, default is "identity".
            cost_function (str): Cost function. Choices is "mse", "logit". 
                Optional, default "mse".
            alpha (float): L2 regularization term. Default is 0.0.

        Raises:
            AssertionError: if input_data_size is not a list.
            AssertionError: if layer_sizes is less than two.
        """

        assert isinstance(layer_sizes, list), "must provide a layer size list"
        assert len(layer_sizes) >= 2, ("Must have at least two layers: "
                                       "len(layer_sizes)={}".format(
                                           len(layer_sizes)))

        self._set_layer_activation(activation)
        self._set_final_layer_activation(final_activation)
        self._set_cost_function(cost_function)

        # L2 regularization term
        self.alpha = alpha

        # Sets up weights and biases
        self.weights = [
            np.random.rand(l_j, l_i)
            for l_i, l_j in zip(layer_sizes[:-1], layer_sizes[1:])]

        # for i, w in enumerate(self.weights):
        #     print(i, w.shape)

        self.biases = [np.random.rand(l_j, 1) for l_j in layer_sizes[1:]]

        self.layer_sizes = layer_sizes
        self.N_layers = len(layer_sizes)

    def _set_layer_activation(self, activation):
        """Sets the layer activation."""
        if activation == "sigmoid" or "logistic":
            self._activation = sigmoid
            self._activation_derivative = sigmoid_derivative
        elif activation == "identity":
            self._activation = identity
            self._activation_derivative = identity_derivative
        # elif activation == "relu":
        #     self._activation =
        #     self._activation_derivative =
        # elif activation == "tanh":
        #     self._activation =
        #     self._activation_derivative =
        # elif activation == "heaviside":
        #     self._activation =
        #     self._activation_derivative =
        else:
            raise ValueError("Activation type '{}' not recognized.".format(
                activation))

    def _set_final_layer_activation(self, final_activation):
        """Sets the final layer activation."""
        if final_activation == "sigmoid" or "logistic":
            self._final_activation = sigmoid
            self._final_activation_derivative = sigmoid_derivative
        elif final_activation == "identity":
            self._final_activation = identity
            self._final_activation_derivative = identity_derivative
        # elif final_activation == "relu":
        #     self._final_activation =
        #     self._final_activation_derivative =
        # elif final_activation == "tanh":
        #     self._final_activation =
        #     self._final_activation_derivative =
        # elif final_activation == "heaviside":
        #     self._final_activation =
        #     self._final_activation_derivative =
        # elif final_activation == "softmax":
        #     self._final_activation =
        #     self._final_activation_derivative =
        else:
            raise ValueError(("Final layer activation type '{}' not "
                              "recognized.".format(activation)))

    def _set_cost_function(self, cost_function):
        """Sets the cost function to use."""
        if cost_function == "mse":
            self._base_cost_function = mse_cost
            self._base_cost_function_derivative = mse_cost_derivative
        else:
            raise ValueError("Cost function '{}' not recognized.".format(
                cost_function))

    def _cost_function(self, x, y):
        """Cost function"""
        base_cost = self._base_cost_function(x, y)

        # L2 regularization
        if alpha != 0.0:
            l2_cost = np.linalg.norm(x - y)
        else:
            l2_cost = 0

        return base_cost + self.alpha*l2_cost

    def _cost_function_derivative(self, x, y):
        """Derivative of the cost function."""
        base_cost_derivative = self._base_cost_function_derivative(x, y)

        # L2 regularization
        if self.alpha != 0.0:
            l2_cost_derivative = 2*(x-y)
        else:
            l2_cost_derivative = 0

        return base_cost_derivative + self.alpha*l2_cost_derivative

    def predict(self, x):
        return self._forward_pass(x)[-1]

    def _forward_pass(self, activation):
        """Performs a feed-forward to the last layer."""
        activations = [activation]
        for i in range(self.N_layers - 1):
            z = (self.weights[i] @ activations[i])
            z += self.biases[i]

            if i+1 != (self.N_layers - 1):
                activations.append(self._activation(z))

        activations.append(self._final_activation(z))

        return activations

    def back_propagate(self, x, y):
        """Performs back-propagation on a single dataset.

        Args:
            x (ndarray): initial layer input.
            y (ndarray): true output values(labels).

        Returns:
            (list(ndarray)): all layer weight gradients
            (list(ndarray)): all layer bias gradients
        """

        # print(x.shape, y.shape, self.weights[0].shape, self.biases[0].shape)

        # Retrieves the z and sigmoid for each layer in sample
        z_list = []
        self.activations = [x]
        for i in range(self.N_layers - 1):
            z = self.weights[i] @ self.activations[i]
            z += self.biases[i]
            z_list.append(z)

            if (i+1) != (self.N_layers - 1):
                # Middle layer activation
                self.activations.append(self._activation(z))
            else:
                # Sigmoid output layer
                self.activations.append(self._final_activation(z).T)

        # Backpropegates
        self.delta_w = [np.empty(w.shape) for w in self.weights]
        self.delta_b = [np.empty(b.shape) for b in self.biases]

        # Gets initial delta value, first of the four equations
        delta = self._cost_function_derivative(self.activations[-1], y).T
        # delta *= self._final_activation_derivative(z_list[-1])

        # Sets last element before back-propagating
        self.delta_b[-1] = delta
        self.delta_w[-1] = delta @ self.activations[-2].T

        # Loops over layers
        for l in range(2, self.N_layers):
            # Second equation: delta^l = delta^{l+1} W^l * dsigma(z^l)
            # delta = (self.weights[-l+1].T @ delta) * \
            #     sigmoid_derivative(z_list[-l])
            z = z_list[-l]
            sp = self._activation_derivative(z)

            # Sets up delta^l
            delta = self.weights[-l+1].T @ delta
            delta *= sp

            self.delta_b[-l] = delta  # np.sum(delta, axis=1)
            self.delta_w[-l] = delta @ self.activations[-l-1].T
            # self.delta_b[-l] = np.mean(delta, axis=0)
            # self.delta_w[-l] /= x.shape[0]

        # print(self.delta_b[0].sum())

        return self.delta_w, self.delta_b

    def train(self, data_train, data_train_labels, epochs=5,
              mini_batch_size=50, eta=1.0, data_test=None,
              data_test_labels=None):
        """Trains the neural-net on provided data. Assumes data size 
        is the same as what provided in the initialization.

        Uses Stochastic Gradient Descent, SGA, and mini-batches to get the 
        deed done.

        Args:
            data_train (ndarray): training data. Shape: [training_sets, 
                set_size]
            data_train_labels (ndarray): training data labels.
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

        # for epoch in tqdm(range(epochs), "Epoch"):
        for epoch in range(epochs):

            # Performs the SGA step of shuffling data
            shuffle_indexes = np.random.choice(list(range(N_train_size)),
                                               size=N_train_size, 
                                               replace=False)

            shuffled_data = cp.deepcopy(data_train[shuffle_indexes])
            shuffled_labels = cp.deepcopy(data_train_labels[shuffle_indexes])

            # print("Origina data:")
            # print (data_train.shape)
            # print (data_train_labels.shape)
            # print("Shuffeld data:")
            # print(shuffled_data.shape)
            # print(shuffled_labels.shape)
            # print("Shuffled indices")
            # print(shuffle_indexes.shape)

            shuffled_data = [
                shuffled_data[i:i+mini_batch_size]
                for i in range(0, N_train_size, number_batches)]
            shuffled_labels = [
                shuffled_labels[i:i+mini_batch_size]
                for i in range(0, N_train_size, number_batches)]

            # print("After splitting, first element:")
            # print(shuffled_data[0].shape)
            # print(shuffled_labels[0].shape)
            # print(shuffle_indexes[0].shape)

            for mb_data, mb_labels in zip(shuffled_data, shuffled_labels):

                # print ("First element of for-loop:")
                # print(mb_data.shape, mb_labels.shape)
                # exit(1)

                self.update_mini_batch(mb_data, mb_labels, eta)

            # print(self.weights[1][:5])

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
            delta_w, delta_b = self.back_propagate(sample, label)

            delta_w_sum = [dw + dws for dw, dws in zip(delta_w, delta_w_sum)]
            delta_b_sum = [db + dbs for db, dbs in zip(delta_b, delta_b_sum)]
            # print((delta_w_sum[0]*eta/len(mb_data)).sum())
            # exit(1)

        # Updates weights and biases
        for l in range(self.N_layers-1):
            self.weights[l] -= (delta_w_sum[l]*eta/len(mb_data))
            self.biases[l] -= (delta_b_sum[l]*eta/len(mb_data))

    def evaluate(self, test_data, test_labels):
        """Evaluates test data."""
        results = []
        for test, label in zip(test_data, test_labels):
            pred = self.predict(np.atleast_2d(test))
            results.append(int(np.argmax(pred) == np.argmax(label)))

            # plot_image(test, np.argmax(label), np.argmax(pred))

        # exit(1)

        return sum(results)


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

    # Converts data to ((N, p-1)) shape
    data_train_samples = np.asarray(
        [d_.reshape((-1, 1)) for d_ in data_train[0]])
    data_valid_samples = np.asarray(
        [d_.reshape((-1, 1)) for d_ in data_valid[0]])
    data_test_samples = np.asarray(
        [d_.reshape((-1, 1)) for d_ in data_test[0]])

    # Converts labels from single floats to arrays with 1.0 at correct output
    data_train_labels = np.asarray(
        [convert_output(l, 10) for l in data_train[1]])
    data_valid_labels = np.asarray(
        [convert_output(l, 10) for l in data_valid[1]])
    data_test_labels = np.asarray(
        [convert_output(l, 10) for l in data_test[1]])

    # print([data_train_samples.shape, 8, 10])
    MLP = MultilayerPerceptron([data_train_samples.shape[1], 50, 10],
                               final_activation="sigmoid")
    MLP.train(data_train_samples, data_train_labels,
              data_test=data_test_samples, 
              data_test_labels=data_test_labels,
              mini_batch_size=20,
              epochs=10,
              eta=1.0)
    print(MLP.evaluate(data_test_samples, data_test_labels))


def __test_nn_sklearn_comparison():
    import warnings
    import copy as cp
    from sklearn.neural_network import MLPRegressor

    # Basic 1-3-3-1 test
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 2])

    # # 2-3-3-2 test
    X = np.array([[0.0, 0.5], [1.0, 1.5]])
    y = np.array([0, 1.0])

    # 10-20-20-10 test
    X = np.random.rand(100,10)
    y = np.random.rand(100)

    mlp = MLPRegressor(
        solver='sgd',               # Stochastic gradient descent.
        activation='logistic',      # Skl name for sigmoid.
        alpha=0.0,                  # No regularization for simplicity.
        # hidden_layer_sizes=(3, 3))  # Full network is of size (1,3,3,1).
        hidden_layer_sizes=(20, 20))  # Full network is of size (1,3,3,1).
    mlp.out_activation_ = "logistic"

    # Force sklearn to set up all the necessary matrices by fitting a data set.
    # We dont care if it converges or not, so lets ignore raised warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp.fit(X, y)

    # A single, completely random, data point which we will propagate through
    # the network.
    X = np.array([[1.125982598]])
    target = np.array([8.29289285])

    X = np.array([[1.125982598, 2.937172838]])
    target = np.array([8.29289285])

    X = np.array([np.random.rand(10)])
    target = np.array([8.29289285])

    # ==========================================================================
    n_samples, n_features = X.shape
    batch_size = n_samples
    hidden_layer_sizes = mlp.hidden_layer_sizes
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = ([n_features] + hidden_layer_sizes + [mlp.n_outputs_])
    activations = [X]
    activations.extend(np.empty((batch_size, n_fan_out))
                       for n_fan_out in layer_units[1:])
    deltas = [np.empty_like(a_layer) for a_layer in activations]
    coef_grads = [np.empty((n_fan_in_, n_fan_out_))
                  for n_fan_in_, n_fan_out_ in zip(layer_units[:-1],
                                                   layer_units[1:])]
    intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]
    # ==========================================================================

    mlp.out_activation_ = "logistic"
    activations = mlp._forward_pass(activations)
    loss, coef_grads, intercept_grads = mlp._backprop(
        X, target, activations, deltas, coef_grads, intercept_grads)

    # nn = NeuralNetwork( inputs          = 1,
    #                     hidden_layers   = 2,
    #                     neurons         = 3,
    #                     outputs         = 1,
    #                     activations     = 'sigmoid',
    #             out_activations = 'identity',
    #             cost_function   = 'mse')

    # nn = MultilayerPerceptron([1, 3, 3, 1])
    # nn = MultilayerPerceptron([2, 3, 3, 1], final_activation="identity")
    nn = MultilayerPerceptron([10, 20, 20, 10], final_activation="logistic")
    # print(mlp.out_activation_)
    # exit(1)
    # Copy the weights and biases from the scikit-learn network to your own.
    for i, w in enumerate(mlp.coefs_):
        nn.weights[i] = cp.deepcopy(w.T)
    for i, b in enumerate(mlp.intercepts_):
        nn.biases[i] = cp.deepcopy(b.T.reshape(-1, 1))

    # Call your own backpropagation function, and you're ready to compare with
    # the scikit-learn code.
    y_sklearn = mlp.predict(X)
    y = nn.predict(cp.deepcopy(X).T)

    assert np.allclose(y, y_sklearn), ("Prediction "
                                       "{} != {}".format(y, y_sklearn))

    nn.back_propagate(X.T, target)

    for i, a in enumerate(nn.activations):
        print(i, a.T, activations[i])
        assert np.allclose(a.T, activations[i]), "error in layer {}".format(i)
    else:
        print("Activations are correct.")

    for i, derivative_bias in enumerate(nn.delta_b):
        print(i, derivative_bias.T, intercept_grads[i])
        assert np.allclose(
            derivative_bias.T, intercept_grads[i]), (
            "error in layer {}".format(i))
    else:
        print("Biases derivatives are correct.")

    for i, derivative_weight in enumerate(nn.delta_w):
        print(i, derivative_weight.T, coef_grads[i])
        assert np.allclose(derivative_weight.T,
                           coef_grads[i]), "error in layer {}".format(i)
    else:
        print("Weight derivatives are correct.")

    # print (derivative_weight.T[0])
    # print (coef_grads[0])

if __name__ == '__main__':
    __test_mlp_mnist()
    # __test_nn_sklearn_comparison()
