import numpy as np

# TODO: Implement a multilayer perceptron neural network here!


def sigmoid(z):
    return 1/(1 + np.exp(-z))


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
            np.zeros((l_i, l_j)) 
            for l_i, l_j in zip(layer_sizes[:-1], layer_sizes[1:])]

        self.biases = [
            np.zeros((layer_sizes[l_i], 1))
            for l_i in range(1, len(layer_sizes))]

        for w in self.weights:
            print ("w shape: ", w.shape)

        self.layer_sizes = layer_sizes
        self.N_layers = len(layer_sizes)

    def feed_forward(self, a):
        """Performs a feed-forward to the last layer."""
        for l in range(self.N_layers):
            a = sigmoid(self.weights[l].T @ a + self.biases[l])
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
        z_list = [self.weights[0].T @ x + self.biases[0]]
        sigmoid_list = [sigmoid(z_list[0])]
        for l in range(1, self.N_layers-1):
            z_list.append(self.weights[l].T @ z_list[l-1] + self.biases[l])
            sigmoid_list.append(sigmoid(z_list[-1]))

        # Backpropegates
        delta_w = [np.zeros(w.size) for w in self.weights]
        delta_b = [np.zeros(b.size) for b in self.biases]

        # Gets delta 
        delta = 2*(sigmoid_list[-1] - y) * sigmoid_derivative(z_list[-1])
        delta_w[-1] = np.dot(delta, sigmoid_list[-2].T)
        delta_b[-1] = delta

        for l in range(2, self.N_layers):
            print(l)
            delta = np.dot(self.weights[-l+1].T, delta) * sigmoid_derivative(z_list[-1])
            delta_w[l] = np.dot(delta, sigmoid_list(-l-1).T)
            delta_b[l] = delta
            print(delta.shape, sigmoid_list[1-l].shape, delta_w[-l].shape)

        print(len(delta_w), delta_w[0].shape)
        exit(1)
        return delta_w, delta_b


    def train(self, data_train, data_train_labels, epochs=10, 
        mini_batch_size=100, eta=1.0):
        """Trains the neural-net on provided data. Assumes data size 
        is the same as what provided in the initialization.

        Args:
            data_train (ndarray): training data. Shape: [training_sets, set_size]
            data_train_labels (ndarray): training data labels.
        """

        assert self.layer_sizes[0] == data_train.shape[1]

        number_batches = data_train.shape[0] // mini_batch_size

        # TEMP
        batch_data = data_train[:mini_batch_size]
        batch_labels = data_train_labels[:mini_batch_size]

        for epoch in range(epochs):

            # Resets gradient sums
            delta_w_sum = []
            delta_b_sum = []

            for i_batch, vals_ in enumerate(zip(batch_data, batch_labels)):
                batch, label = vals_
                y_ = np.zeros((self.layer_sizes[-1]))
                y_[label] = 1.0

                delta_w, delta_b = self.back_propegate(batch, y_)

                delta_w_sum.append(delta_w)
                delta_b_sum.append(delta_b)

            # delta_w_sum = np.asarray(delta_w_sum)
            # delta_b_sum = np.asarray(delta_b_sum)

            print (delta_w_sum[-1][-2].shape, delta_b_sum[-1][-1].shape)
            exit(1)
            # Updates weights and biases
            for l in range(self.N_layers):
                self.weights[l] -= eta*np.sum(delta_w_sum)
                self.biases[l] -= eta*np.sum(delta_b_sum)



            raise SystemExit("Good so far!")
                

            print ("Breaks after 1 epoch @ 108")
            break


def __test_mlp():
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

    MLP = MultilayerPerceptron([data_train[0].shape[1], 8, 10])
    MLP.train(data_train[0], data_train[1])


if __name__ == '__main__':
    __test_mlp()
