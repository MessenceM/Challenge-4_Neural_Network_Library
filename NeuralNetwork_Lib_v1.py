"""
Neural Network Library, coded from scratch
Author : Matis Messence
25/06/2023

I am on a coding adventure to learn python, data science and machine learning.
Follow my progress at
https://www.linkedin.com/in/matis-messence/
https://github.com/MessenceM

The Neural Network class takes as argument "shape" an array of the number of nodes in each layer (including
input and output layers) e.g.: [3, 2, 1] means 3 inputs, 2 nodes in the hidden layer, and 1 output node

Other activations functions are implemented as static method but the most effective one was the tanh in my example
Later versions will implement the choice of activation function per layer and choice of loss function as well.
"""
import numpy as np

class NeuralNetwork:
    def __init__(self, shape, lr=0.1):
        # Inputs are in the shape of an array to allow having the desired amount of layers
        self.nb_layers = len(shape)
        self.shape = shape
        self.lr = lr
        self.weights = []
        self.biases = []
        self.A = []
        self.Z = []
        self.accuracy = []
        self.guesses = []
        self.epoch = 0
        # Randomly set the weights in matrix for each layer and the biases

        self.biases = [np.random.randn(j, 1) for j in shape[1:]]
        self.weights = [np.random.randn(j, i) for i, j in zip(shape[:-1], shape[1:])]

    def feedforward(self, a):
        a = np.ndarray((self.shape[0], 1), buffer=a)
        for b, w in zip(self.biases, self.weights):
            a = self.tan_h(np.dot(w, a) + b)
        return a

    def train(self, inputs, targets, epochs):
        # Realize the gradient descent for the batch specified as input
        # In this case the batch is of size 1
        for epoch in range(epochs):
            # Compute the output once
            self.backpropagation(inputs[epoch], targets[epoch])

            # Every 10 epoch, the accuracy of the nn is measured
            if epoch % 1 == 0:
                s = "Epoch: {}, Accuracy: {}".format(epoch, self.get_accuracy())
                print(s)

    def backpropagation(self, a, target):
        # Variables storing for each layer the biases gradient vector and the weight gradients matrix
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        a = np.ndarray((self.shape[0], 1), buffer=a)
        self.A.append(a)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b  # Vertical vector : input of each layer
            a = self.tan_h(z)  # Vertical vector : result of the activation function/output of each layer
            self.A.append(a)
            self.Z.append(z)

        # Get result and accuracy
        result = self.get_result(a)
        self.guesses.append(result)
        acc = 1 if result - target == 0 else 0
        self.accuracy.append(acc)

        # Backprop
        cost = self.d_loss(a, target)  # dC/dA
        print(cost)
        for L in range(1, self.nb_layers):
            sp = self.d_tan_h(self.A[-L])  # dA/dZ Using A instead of Z to avoid computing the tanh again
            delta = cost * sp  # The error at layer L: dC/dZ
            nabla_b[-L] = delta  # Gradient of the biases: dC/dB because dZ/dB = 1 ___ np.sum(dZ, axis=0, keepdims=True)
            nabla_w[-L] = np.dot(delta, self.A[-(L + 1)].T)  # Gradient of the weights
            cost = np.dot(self.weights[-L].T, delta)  # Updates the cost for the other layers

        # Update weights and biases
        for i in range(self.nb_layers - 1):
            self.weights[i] -= self.lr * nabla_w[i]
            self.biases[i] -= self.lr * nabla_b[i]

    def get_accuracy(self):
        return np.mean(self.accuracy)

    @staticmethod
    def get_result(guess):
        # Step function
        return 1 if guess >= 0 else -1

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def d_sigmoid(a):  # The sigmoid has already been computed, so we use the value a
        return a * (1 - a)

    @staticmethod
    def tan_h(z):
        return np.tanh(z)

    @staticmethod
    def d_tan_h(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def d_loss(guess, target):
        return guess - target

