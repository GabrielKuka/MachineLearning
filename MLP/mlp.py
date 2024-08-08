import numpy as np
import random, time

class MLP(object):
    """ Multi Layer Perceptron """

    def __init__(self, num_inputs=3, hidden_l=[3, 3], num_outputs=2):

        self.num_inputs = num_inputs
        self.hidden_l = hidden_l
        self.num_outputs = num_outputs

        # Create a representation of the layers
        layers = [num_inputs] + hidden_l + [num_outputs]

        # Create random weights for the layers
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

        # Create random biases for the layers
        self.biases = []
        for i in range(1, len(layers)):
            b = np.random.rand(layers[i])
            self.biases.append(b)

        # Init derivatives_w: dC/dw
        self.derivatives_w = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            self.derivatives_w.append(d)

        # Init derivatives_w: dC/db
        self.derivatives_b = []
        for i in range(1, len(layers)):
            d = np.zeros(layers[i])
            self.derivatives_b.append(d)

        # Init default activations
        self.activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            self.activations.append(a)

    def forward_propagate(self, inputs):
        """ Important equations:
            net inputs: z = dot(weights, inputs) + bias
            activations = sigmoid(z)
        """
        current_activations = inputs

        self.activations[0] = current_activations

        for i, w  in enumerate(self.weights):
            # dot product
            z = np.dot(current_activations, w) + self.biases[i]

            # Calculate new activations
            current_activations = self._sigmoid(z)

            # Store current activations
            self.activations[i+1] = current_activations

        return current_activations


    def back_propagate(self, cost):
        """ Important equations:
            1. Output layer error = dC/da * sigmoid_derivative
            2. Error at any layer = dot(weights, error at next layer) * sigmoid_derivative
            3. Derivative of Cost with respect to bias = error at that layer
            4. Derivative of Cost with respect to weights = dot(activations, error
               at next layer)
        """

        # Iterate backwards through the network
        for i in reversed(range(len(self.derivatives_w))):

            # Activations of the previous layers
            activations = self.activations[i+1]

            # Find the error at the layer = dC/da * sigmoid_derivative
            error = cost * self._sigmoid_der(activations)

            # Reshape error as a 2d array
            error_reshaped = error.reshape(error.shape[0], -1).T

            # Activations of the current layer
            current_activations = self.activations[i]

            # Reshape it as a 2d column matrix
            current_activations = \
            current_activations.reshape(current_activations.shape[0],
                                        -1)

            # Save the derivatives_w (change of Cost with respect to the
            # weights or dC/dw)
            self.derivatives_w[i] = np.dot(current_activations, error_reshaped)

            # Save the derivatives_b (change of Cost with respect to biases or
            # dC/db)
            self.derivatives_b[i] = error

            # Backpropagate error
            cost = np.dot(error, self.weights[i].T)

    def train(self, inputs, targets, epochs, l_r):

        for i in range(epochs):
            sum_error = 0

            # Iterate through training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # Feed forward
                output = self.forward_propagate(input)

                # Find the cost for each output neuron
                cost = target - output
               # print('[Output: {}\tCost: {}]'.format(output, cost))
                # Back propagate
                self.back_propagate(cost)

                # Update the weights (gradient descent)
                self.gradient_descent(l_r)

                # Keep track of the Mean Squared Error
                sum_error += self._mse(target, output)

            # Display the training error
            print("Error at epoch {}: {}".format(i+1, sum_error / len(inputs)))

        print("Training Completed!")


    def gradient_descent(self, learning_rate=1):
        """ Update the weights and biases """

        # Iterate through each set of weights
        for i in range(len(self.weights)):
            weights = self.weights[i]
            biases = self.biases[i]

            derivatives_w = self.derivatives_w[i]
            derivatives_b = self.derivatives_b[i]

            weights += derivatives_w * learning_rate
            biases += derivatives_b * learning_rate


    def _mse(self, target, output):

        return np.average((target - output) ** 2)

    def _sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))


    def _sigmoid_der(self, x):
        return x * (1.0 - x)


if __name__ == "__main__":

    # Create data to train the network to calculate smth
    inputs = np.array([[random.random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # Create a neural network
    neural_network = MLP(2, [3], 1)

    # Train the network
    start = time.time()
    neural_network.train(inputs, targets, 100, 1.5)
    end = time.time()

    print("Training finished in {} seconds".format(end-start))

    # Test the network
    inputs = np.array([0.6, 0.1])

    output = neural_network.forward_propagate(inputs)

    print("{} + {} = {}".format(inputs[0], inputs[1], output[0]))

