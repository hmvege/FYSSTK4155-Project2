import numpy as np
from tqdm import tqdm

# TODO: Implement a multilayer perceptron neural network here!


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_derivative(z):
    exp_ = np.exp(-z)
    return exp_ / (1 + exp_)**2


class MultilayerPerceptron:
    def __init__(self, layer_sizes):
        """Initializer for multilayer perceptron.

        Number of layers is always minimum N_layers + 2.

        Args:
            layer_sizes (list(int)): list of layer sizes after input data.
                Constists of [input_layer_size, N layer sizes, output_layer].
            input_data_size (int): size of input data.

        Raises:
            AssertionError: if input_data_size is not a list.
            AssertionError: if layer_sizes is less than two.
        """

        assert isinstance(layer_sizes, list), "must provide a layer size list"
        assert len(layer_sizes) >= 2, ("Must have at least two layers: "
                                       "len(layer_sizes)={}".format(
                                           len(layer_sizes)))

        # Sets up weights and biases
        self.weights = [
            np.random.rand(l_j, l_i)
            for l_i, l_j in zip(layer_sizes[:-1], layer_sizes[1:])]

        self.biases = [np.random.rand(l_j, 1) for l_j in layer_sizes[1:]]
        self.layer_sizes = layer_sizes
        self.N_layers = len(layer_sizes)

    def _cost_function(self, a, y):
        """Cost function"""
        return np.sum((a - y)**2, axis=0)/(2*y.shape[0])

    def _cost_function_derivative(self, a, y):
        """Derivative of the cost function."""
        return 2*(a - y)

    def forward_pass(self, a):
        """Performs a feed-forward to the last layer."""
        for l in range(self.N_layers-1):
            a = sigmoid((self.weights[l] @ a) + self.biases[l])
        return a

    def back_propegate(self, x, y):
        """Performs back-propagation on a single dataset.

        Args:
            x (ndarray): initial layer input.
            y (ndarray): true output values(labels).

        Returns:
            (list(ndarray)): all layer weight gradients
            (list(ndarray)): all layer bias gradients
        """

        # Retrieves the z and sigmoid for each layer in sample
        z_list = []
        self.activations = [x.reshape((-1,1))]
        for l in range(0, self.N_layers-1):
            # print(self.weights[l].shape, activations[-1].shape, self.biases[l].shape)
            z = (self.weights[l] @ self.activations[-1]).reshape((-1,1)) 
            z += self.biases[l]
            z_list.append(z)
            self.activations.append(sigmoid(z_list[l]))

        # Backpropegates
        self.delta_w = [np.zeros(w.shape) for w in self.weights]
        self.delta_b = [np.zeros(b.shape) for b in self.biases]

        # Gets initial delta value, first of the four equations
        delta = self._cost_function_derivative(self.activations[-1], y)
        delta *= sigmoid_derivative(z_list[-1])

        # Sets last element before back-propagating
        self.delta_b[-1] = delta
        self.delta_w[-1] = delta @ self.activations[-2].T

        # Loops over layers
        for l in range(2, self.N_layers):
            # Second equation: delta^l = delta^{l+1} W^l * dsigma(z^l)
            delta = (self.weights[-l+1].T @ delta) * \
                sigmoid_derivative(z_list[-l])
            self.delta_b[-l] = delta
            self.delta_w[-l] = delta @ self.activations[-l-1].T

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

        assert self.layer_sizes[0] == data_train.shape[1], ("training data "
            "and labels do not match in shape: {} != {}".format(
                self.layer_sizes[0], data_train.shape[1]))

        # Sets if we are to evaluate the data while running
        if (not isinstance(data_test, type(None))) and \
            (not isinstance(data_test_labels, type(None))):
            perform_eval = True
        else:
            perform_eval = False

        # Gets the number of batches
        number_batches = data_train.shape[0] // mini_batch_size

        # print (batch_data)

        # exit(1)
        for epoch in tqdm(range(epochs),"Epoch"):
            c=0
            # print(epoch,  self.weights)
            np.random.shuffle(data_train)

            # Splits into batches
            batch_data = [
                data_train[i*mini_batch_size:mini_batch_size*(i+1)]
                for i in range(number_batches)]
            batch_labels = [
                data_train_labels[i*mini_batch_size:mini_batch_size*(i+1)]
                for i in range(number_batches)]

            # Loops over minibatches
            for mb_data, mb_labels in zip(batch_data, batch_labels):
                c+=1

                # Resets gradient sums
                delta_w_sum = [np.zeros(w.shape) for w in self.weights]
                delta_b_sum = [np.zeros(b.shape) for b in self.biases]

                # Loops over all samples and labels in mini batch
                for sample, label in zip(mb_data, mb_labels):

                    # import matplotlib.pyplot as plt
                    # from matplotlib import cm
                    # plt.imshow(sample.reshape(int(np.sqrt(sample.shape[0])), int(np.sqrt(sample.shape[0]))), cmap=cm.gray)
                    # plt.show()

                    # Sets up output vector
                    y_ = np.zeros((self.layer_sizes[-1], 1))
                    y_[label] = 1.0

                    delta_w, delta_b = self.back_propegate(sample, y_)

                    delta_w_sum = [dws + dw for dws, dw in zip(delta_w_sum, delta_w)]
                    delta_b_sum = [dbs + db for dbs, db in zip(delta_b_sum, delta_b)]

                # print( self.weights)
                print()
                print(len(delta_w_sum), self.N_layers, delta_w_sum[0].shape, self.weights[0].shape)
                print(delta_w_sum[0].shape, delta_w_sum[0][0])
                exit(1)

                # Updates weights and biases
                print (self.weights[0][0,0], delta_w_sum[0]*eta/float(mini_batch_size))

                self.weights = [w - dw*eta/float(mini_batch_size) for w, dw in zip(self.weights, delta_w_sum)]
                self.biases = [b - db*eta/float(mini_batch_size) for b, db in zip(self.biases, delta_b_sum)]

                print (self.weights[0][0,0])
                print(self.weights[0].shape, delta_w_sum[0].shape)
                    # print(np.sum(temp_w_sum)/float(mini_batch_size))
                    # if l==1:exit(1)

                # # print( self.weights)
                if c==3: exit(1)


            if perform_eval:
                print("Epoch: {} Score: {}/{}".format(
                    epoch, self.evaluate(data_test, data_test_labels), 
                    len(data_test_labels)))


    def evaluate(self, test_data, test_labels):
        """Evaluates test data."""
        results = []
        for test, label in zip(test_data, test_labels):
            results.append(np.argmax(self.forward_pass(test))==label)
        # print (sum(results))
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

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 2])

    MLP = MultilayerPerceptron([data_train[0].shape[1], 8, 10])
    MLP.train(data_train[0][:10000], data_train[1][:10000], 
        data_test=data_test[0], data_test_labels=data_test[1])
    print(MLP.evaluate(data_test[0], data_test[1]))

def __test_nn_sklearn_comparison():
    import warnings
    from sklearn.neural_network import MLPRegressor

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 2])
    mlp = MLPRegressor( solver              = 'sgd',      # Stochastic gradient descent.
                        activation          = 'logistic', # Skl name for sigmoid.
                        alpha               = 0.0,        # No regularization for simplicity.
                        hidden_layer_sizes  = (3, 3) )    # Full network is of size (1,3,3,1).

    # Force sklearn to set up all the necessary matrices by fitting a data set. 
    # We dont care if it converges or not, so lets ignore raised warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp.fit(X,y)

    # A single, completely random, data point which we will propagate through 
    # the network.
    X      = np.array([[1.125982598]])
    target = np.array([ 8.29289285])
    mlp.predict(X)

    # ==========================================================================
    n_samples, n_features   = X.shape
    batch_size              = n_samples
    hidden_layer_sizes      = mlp.hidden_layer_sizes
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = ([n_features] + hidden_layer_sizes + [mlp.n_outputs_])
    activations = [X]
    activations.extend(np.empty((batch_size, n_fan_out)) 
                       for n_fan_out in layer_units[1:])
    deltas      = [np.empty_like(a_layer) for a_layer in activations]
    coef_grads  = [np.empty((n_fan_in_, n_fan_out_)) 
                   for n_fan_in_, n_fan_out_ in zip(layer_units[:-1],
                                                    layer_units[1:])]
    intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]
    # ==========================================================================

    activations                       = mlp._forward_pass(activations) 
    loss, coef_grads, intercept_grads = mlp._backprop(
            X, target, activations, deltas, coef_grads, intercept_grads)

    # nn = NeuralNetwork( inputs          = 1,
    #                     hidden_layers   = 2,
    #                     neurons         = 3,
    #                     outputs         = 1,
    #                     activations     = 'sigmoid',
    #             out_activations = 'identity',
    #             cost_function   = 'mse')

    nn = MultilayerPerceptron([1,3,3,1])

    # Copy the weights and biases from the scikit-learn network to your own.
    for i, w in enumerate(mlp.coefs_):
        print(i, w.shape, nn.weights[i].shape)
        nn.weights[i] = w.T
    for i, b in enumerate(mlp.intercepts_):
        nn.biases[i]  = b.T.reshape(-1,1)

    # Call your own backpropagation function, and you're ready to compare with 
    # the scikit-learn code.
    print(nn.weights[0].shape)
    y = nn.forward_pass(X)
    nn.back_propegate(y, target)




    for i, a in enumerate(nn.activations) :
        print(a, activations[i])
        assert np.allclose(a, activations[i])

    for i, derivative_bias in enumerate(nn.derivative_biases) :
        assert np.allclose(derivative_bias, intercept_grads[i])

    for i, derivative_weight in enumerate(nn.derivative_weights) :
        assert np.allclose(derivative_weight, coef_grads[i])



if __name__ == '__main__':
    # __test_mlp_mnist()
    __test_nn_sklearn_comparison()
