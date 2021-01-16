from sklearn.datasets import fetch_openml
from sklearn import preprocessing
import sklearn as skl
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# data processing

x = (x/255).astype('float32')
y = y.reshape(-1, 1)
y = y.astype(np.float)
ohe = preprocessing.OneHotEncoder()
y = ohe.fit_transform(y).toarray()

x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y, test_size=0.15, random_state=42)

x_train, x_test = x_train.T, x_test.T
y_train, y_test = y_train.T, y_test.T



class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=100, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.layers = len(sizes)
        # save all parameters in the neural network in this dictionary:
        self.p = self.initialisation(sizes)
        self.cost_list = np.zeros(self.epochs)

    # changing parameter

    def cp(self, a, i):
        return a + str(i)

    def np(self, a, i):
        return a + str(i + 1)

    def pp(self, a, i):
        return str(a) + str(i - 1)

    # initialisation

    def initialisation(self, sizes):
        self.p = {}
        # initialise weight and bias
        for i in np.arange(1, self.layers):
            self.p[self.cp('W', i)] = np.random.randn(sizes[i], sizes[i-1]) * np.sqrt(1. / sizes[i-1])
            self.p[self.cp('B', i)] = np.zeros((sizes[i], 1))
            # print('W' + str(i) + ', s = ' + str(self.p[self.cp('W', i)].shape))
        # initialise bias
        return self.p

    # required equations

    def linear_forward(self, w, a, b):
            Z = np.dot(w, a) + b
            return Z

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    # feed forward

    def forward_pass(self, x_train):
        # generate initial input
        self.p['A0'] = x_train
        x_train = x_train.T
        # forward pass middle layers
        for i in np.arange(1, self.layers - 1):
            w, a, b = self.p[self.cp('W', i)], self.p[self.pp('A', i)], self.p[self.cp('B', i)]
            self.p[self.cp('Z', i)] = self.linear_forward(w, a, b)
            self.p[self.cp('A', i)] = self.sigmoid(self.p[self.cp('Z', i)])
        # forward pass softmax
        for i in np.arange(self.layers-1, self.layers):
            w, a, b, = self.p[self.cp('W', i)], self.p[self.pp('A', i)], self.p[self.cp('B', i)]
            self.p[self.cp('Z', i)] = self.linear_forward(w, a, b)
            self.p[self.cp('A', i)] = self.softmax(self.p[self.cp('Z', i)])
        return self.p

    # compute cost

    def compute_cost(self, output, Y):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(output)) + np.multiply(1 - Y, np.log(1 - output)))
        return cost

    # backwards propagation

    def linear_backward(self, dz, params):
        a_prev, w, b = params
        m = a_prev.shape[1]
        dw = (1. / m) * np.dot(dz, a_prev.T)
        db = (1. / m) * np.sum(dz, axis=1, keepdims=True);
        da_prev = np.dot(w.T, dz)
        return da_prev, dw, db

    def backward_pass(self, y_train):
        # generate output error (dA)
        output = self.p[self.cp('A', self.layers-1)]
        self.p[self.cp('dA', self.layers-1)] = - (np.divide(y_train, output) - np.divide(1 - y_train, 1 - output))
        # backwards softmax
        for i in reversed(np.arange(self.layers-1, self.layers)):
            params = self.p[self.pp('A', i)], self.p[self.cp('W', i)], self.p[self.cp('B', i)]
            self.p[self.cp('dZ', i)] = self.softmax(self.p[self.cp('dA', i)], derivative=True)
            da_prev, dw, db = self.linear_backward(self.p[self.cp('dZ', i)], params)
            self.p[self.pp('dA', i)], self.p[self.cp('dW', i)], self.p[self.cp('dB', i)] = da_prev, dw, db
        # backwards sigmoid
        for i in reversed(np.arange(1, self.layers-1)):
            params = self.p[self.pp('A', i)], self.p[self.cp('W', i)], self.p[self.cp('B', i)]
            self.p[self.cp('dZ', i)] = self.sigmoid(self.p[self.cp('dA', i)], derivative=True)
            da_prev, dw, db = self.linear_backward(self.p[self.cp('dZ', i)], params)
            self.p[self.pp('dA', i)], self.p[self.cp('dW', i)], self.p[self.cp('dB', i)] = da_prev, dw, db
        return self.p

    def update_parameters(self):
        for i in np.arange(1, self.layers - 1):
            self.p[self.cp('W', i)] -= self.l_rate * self.p[self.cp('dW', i)]
            self.p[self.cp('B', i)] -= self.l_rate * self.p[self.cp('dB', i)]
        return self.p

    def compute_accuracy(self, x_test, y_test):
        self.forward_pass(x_test)
        output = np.argmax(self.p['A' + str(self.layers -1)], axis=0)
        labels = np.argmax(y_test, axis=0)
        print(confusion_matrix(output, labels))
        print(classification_report(output, labels))

    def train(self, x_train, y_train):
        for i in range(self.epochs):
            self.forward_pass(x_train)
            self.backward_pass(y_train)
            self.update_parameters()
            self.cost_list[i] = self.compute_cost(self.p['A' + str(self.layers-1)], y_train)


dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])

# list(dnn.p.keys())

