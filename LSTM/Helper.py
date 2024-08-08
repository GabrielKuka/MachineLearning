import numpy as np

class Helper(object):

    @staticmethod
    def sigmoid(input):
        
        result = 1 / (1+np.exp(-input))

        return result

    @staticmethod
    def softmax(input):

        expression = np.exp(input - np.max(input))
        result = expression / np.sum(expression)

        return result

    @staticmethod
    def tanh(input):
        result = np.tanh(input)

        return result