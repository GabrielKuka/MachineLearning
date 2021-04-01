import numpy as np

# Xavier Initialization method
class Xavier:

    def __init__(self, inputs, units):
        self.inputs = inputs
        self.units = units 

        self.std = (1.0/np.sqrt(self.inputs + self.units))

    def get_random_weights(self):
        weights = np.random.randn(self.units, self.inputs + self.units ) * self.std

        return weights

    def get_random_biases(self):
        biases = np.zeros((self.units, 1))

        return biases

    def get_forget_gate_biases(self):
        biases = np.ones((self.units, 1))

    def get_output_weights(self):
        weights = np.random.randn(self.inputs + self.units) * \
            (1/np.sqrt(self.inputs))

        return weights

    def get_output_biases(self):
        biases = np.zeros((self.inputs, 1))

        return biases