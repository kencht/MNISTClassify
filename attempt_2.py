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

data = [x_train, x_test, y_train, y_test]
data = [x.T for x in data]

# class

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=2000, l_rate=1, beta=0.9):
        self.sizes = sizes
        self.beta = beta
        self.epochs = epochs
        self.l_rate = l_rate
        self.layers = len(sizes)
        # save all parameters in the neural network in this dictionary
        self.p = self.initialisation(sizes)
        # save costs
        self.cost_list = np.zeros(self.epochs)

    # parameter change

    def cp(self, a, i):
        return a + str(i)

    def np(self, a, i):
        return a + str(i + 1)

    def pp(self, a, i):
        return str(a) + str(i - 1)

    # initialise weight and bias

    def initialisation(self, sizes):
        self.p = {}
        # initialise weight and bias
        for i in np.arange(1, self.layers):
            self.p[self.cp('W', i)] = np.random.randn(sizes[i], sizes[i-1]) * np.sqrt(1. / sizes[i-1])
            self.p[self.cp('v_dW', i)] = np.zeros(self.p[self.cp('W', i)].shape)
            self.p[self.cp('B', i)] = np.zeros((sizes[i], 1))
            self.p[self.cp('v_dB', i)] = np.zeros(self.p[self.cp('B', i)].shape)
            # print('W' + str(i) + ', s = ' + str(self.p[self.cp('W', i)].shape))
        # initialise bias
        return self.p

    # forward pass equations

    def linear_forward(self, w, a, b):
        Z = np.dot(w, a) + b
        return Z

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def softmax(self, Z):
        e_Z = np.exp(Z - np.max(Z))
        return e_Z / e_Z.sum(axis=0)

    def forward_pass(self, x_train):
        # generate initial input
        self.p['A0'] = x_train
        # forward pass through layers
        self.p['Z1'] = self.linear_forward(self.p['W1'], self.p['A0'], self.p['B1'])
        self.p['A1'] = self.sigmoid(self.p['Z1'])
        self.p['Z2'] = self.linear_forward(self.p['W2'], self.p['A1'], self.p['B2'])
        self.p['A2'] = self.sigmoid(self.p['Z2'])
        self.p['Z3'] = self.linear_forward(self.p['W3'], self.p['A2'], self.p['B3'])
        self.p['A3'] = self.softmax(self.p['Z3'])
        return self.p

    # compute cost

    def compute_cost(self, output, Y):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(output)) + np.multiply(1 - Y, np.log(1 - output)))
        return cost

    # backward pass equations

    def linear_backward(self, dZ, params):
        a_prev, w = params
        m = a_prev.shape[1]
        dW = (1. / m) * np.dot(dZ, a_prev.T)
        dB = (1. / m) * np.sum(dZ, axis=1, keepdims=True);
        dA_prev = np.dot(w.T, dZ)
        return dW, dA_prev, dB

    def sigmoid_backward(self, dA, params):
        Z = params
        s = self.sigmoid(Z)
        dZ = dA * s * (1 - s)
        return dZ

    # backwards pass

    def backward_pass(self, y_train):
        # first weight
        self.p['dZ3'] = self.p['A3'] - y_train
        params = self.p['A2'], self.p['W3']
        self.p['dW3'], self.p['dA2'], self.p['dB3'] = self.linear_backward(self.p['dZ3'], params)
        # second weight
        params = self.p['Z2']
        self.p['dZ2'] = self.sigmoid_backward(self.p['dA2'], params)
        params = self.p['A1'], self.p['W2']
        self.p['dW2'], self.p['dA1'], self.p['dB2'] = self.linear_backward(self.p['dZ2'], params)
        # third weight
        params = self.p['Z1']
        self.p['dZ1'] = self.sigmoid_backward(self.p['dA1'], params)
        params = self.p['A0'], self.p['W1']
        self.p['dW1'], self.p['dA0'], self.p['dB1'] = self.linear_backward(self.p['dZ1'], params)
        return self.p

    # update parameters

    def update_parameters(self):
        for i in np.arange(1, self.layers - 1):
            self.p[self.cp('v_dW', i)] = (self.beta * self.p[self.cp('v_dW', i)] + (1. - self.beta) * self.p[self.cp('dW', i)])
            self.p[self.cp('v_dB', i)] = (self.beta * self.p[self.cp('v_dB', i)] + (1. - self.beta) * self.p[self.cp('dB', i)])
            self.p[self.cp('W', i)] -= self.l_rate * self.p[self.cp('v_dW', i)]
            self.p[self.cp('B', i)] -= self.l_rate * self.p[self.cp('v_dB', i)]
        return self.p

    # compute accuracy

    def compute_accuracy(self, x_test, y_test):
        self.forward_pass(x_test)
        predictions = np.argmax(self.p['A3'], axis=0)
        labels = np.argmax(y_test, axis=0)
        print(confusion_matrix(predictions, labels))
        print(classification_report(predictions, labels))

    # train

    def train(self, x_train, x_test, y_train, y_test):
        for i in range(self.epochs):
            self.forward_pass(x_train)
            self.backward_pass(y_train)
            self.update_parameters()
            self.cost_list[i] = self.compute_cost(self.p['A' + str(self.layers-1)], y_train)
            if (i % 2 == 0):
                print("Epoch", i, "cost: ", self.cost_list[i])
        self.compute_accuracy(x_test, y_test)


dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])