"""
Neural Network Library, coded from scratch
Author : Matis Messence
25/06/2023

I am on a coding adventure to learn python, data science and machine learning.
Follow my progress at
https://www.linkedin.com/in/matis-messence/
https://github.com/MessenceM

Trying to solve the XOR problem through a Neural Network library coded from scratch
I managed to reach 93.5% accuracy with a network of [2, 6, 1] with tanh activation function
"""

from NeuralNetwork_Lib_v1 import NeuralNetwork
from matplotlib import pyplot as plt
import numpy as np


def create_data(nb_points, min, max):
    X = np.random.uniform(min, max, (nb_points, 2))
    y = np.ndarray(nb_points)
    for i in range(nb_points):
        a, b = X[i][0], X[i][1]
        if b >= f(a) and b >= g(a) or b <= f(a) and b <= g(a):
            y[i] = 1
        else:
            y[i] = -1
    return X, y

# Functions to separate the data
def f(x): return 0.2 * x - 0.1
def g(x): return - x + 0.2

# Min and max values of the input
min = -1
max = 1
nb_points = 100000
X, y = create_data(nb_points, min, max)

# The argument is an array of the number of nodes in each layer
nn = NeuralNetwork([2, 6, 1])
nn.train(X, y, nb_points)

# Plot the results
plt.scatter(X[2000:, 0], X[2000:, 1], c=nn.guesses[2000:])  # Last 200 samples
# plt.scatter(X[:, 0], X[:, 1], c=y)  # Labels
plt.plot([min, max], [f(min), f(max)], [min, max], [g(min), g(max)])  # Functions
plt.show()


