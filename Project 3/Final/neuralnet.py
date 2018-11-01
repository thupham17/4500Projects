#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:39:31 2018

@author: issa18
"""

from numpy import exp, random, dot, zeros, sqrt

import sys

def inputfile(filename):
    '''Reads the input file'''

    try:
        f = open(filename, 'r')
    except IOError:
        print ("Cannot open file \'{0}\'\n".format(filename))
        sys.exit("bye")

    # read data
    data = f.readlines()
    f.close()

    # Parse first line
    line0 = data[0].split()
    if len(line0) == 0:
        sys.exit('Empty first line.')


    if filename == 'russell_prices.txt':
        n = int(line0[1])
        m = int(line0[3])
        matrix = zeros((n,m))
        #line1 = data[1].split()
        for i in range(n):
            theline = data[i+2].split()
            for j in range(m):
                valueij = float(theline[j])
                matrix[i][j] = valueij
    return matrix, n, m


class Layer():
    def __init__(self, nodes, node_inputs):
        self.weights = 2 * random.randn(nodes, node_inputs) - 1

class NeuralNetwork():
    def __init__(self, layer1, layer2, layer3):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    def gradient_descent(self, train_input, validation, loops):
        current_error = 10000
        for i in range(loops):

            layer1_out, layer2_out, layer3_out  = self.compute_output(train_input)
            layer3_e = validation.T - layer3_out
            delta3 = layer3_e * self.dsigmoid(layer3_out)

            layer2_e = delta3.dot(self.layer3.weights.T)
            delta2 = layer2_e * self.dsigmoid(layer2_out)
            layer1_e = delta2.dot(self.layer2.weights.T)
            delta1 = layer1_e * self.dsigmoid(layer1_out)

            layer1_descent = train_input.dot(delta1)
            layer2_descent = layer1_out.T.dot(delta2)
            layer3_descent = layer2_out.T.dot(delta3)

            self.layer1.weights += layer1_descent
            self.layer2.weights += layer2_descent
            self.layer3.weights += layer3_descent

            new_error = ((validation.T - layer3_out)**2).mean(axis=None)
            print("Current error is: ", new_error)
            if (current_error - new_error < 0.0000000001):
                break
            else:
                current_error = new_error
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def dsigmoid(self, s):
        return s * (1 - s)

    def compute_output(self, inputs):
        layer1_out = self.sigmoid(dot(inputs.T, self.layer1.weights))
        layer2_out = self.sigmoid(dot(layer1_out, self.layer2.weights.T))
        layer3_out = self.sigmoid(dot(layer2_out, self.layer3.weights))

        return layer1_out, layer2_out, layer3_out


def returns(matrix):
    returns = zeros((matrix.shape))
    m = matrix.shape[1]
    n = matrix.shape[0]
    for i in range(n):
        for j in range(m):
            if j == 0:
                returns[i][j] = 0
            else:
                returns[i][j] = (matrix[i][j] - matrix[i][j-1]) / matrix[i][j-1]
    return returns


Prices, num_assets, num_days = inputfile('russell_prices.txt')
Returns = returns(Prices)
#input and output nodes
n = num_assets
#middle layer nodes
m = 50
#Running the code
random.seed(0)

layer1 = Layer(n, m)
layer2 = Layer(m, m)
layer3 = Layer(m, n)

four_network = NeuralNetwork(layer1, layer2, layer3)
train_inputs = Returns[:,1:243]
train_outputs = Returns[:,11:253]

four_network.gradient_descent(train_inputs, train_outputs, 10000)

test_inputs = Returns[:,253:494]
test_outputs = Returns[:,263:504]
predictions = four_network.compute_output(test_inputs)
predictions = predictions[2]
#test performance on new data
print ("Error on test set: ", ((test_outputs.T - predictions)**2).mean(axis=None))
