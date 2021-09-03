import numpy as np
import pandas as pd


__author__ = "Vidyadhar Mudium"

def cross_entropy_loss(y_pred, y_train):
    y = np.zeros(y_pred.shape)
    y[np.arange(len(y_train)), y_train] = 1
    return np.sum(-y * np.log(y_pred))


def accuracy(y_pred,y_train):
    y_pred = np.argmax(y_pred, axis=1)
    return np.sum(y_pred == y_train) / len(y_train) * 100


class NeuralNetwork(object):
    def __init__(self, input_no, hidden_no, output_no ):
        self.h = np.zeros(hidden_no)
        self.w1 = np.random.uniform(0, 0.01, (input_no, hidden_no))
        self.b1 = np.random.uniform(0, 0.01, hidden_no)
        self.w2 = np.random.uniform(0, 0.01, (hidden_no, output_no))
        self.b2 = np.random.uniform(0, 0.01, output_no)
        pass


    def forward(self, x):
        self.h = np.dot(x, self.w1) + self.b1
        self.h = (np.max([np.zeros(self.h.shape), self.h], axis=0))
        
        y_pred = (np.dot(self.h, self.w2) + self.b2)
        y_pred = np.exp(y_pred)
        y_pred = y_pred/(np.sum(y_pred, axis=1).reshape((len(y_pred), 1)))
        return y_pred


    def backward(self, x, y_train, y_pred, lr):
        y = np.zeros(y_pred.shape)
        y[np.arange(len(y_train)), y_train] = 1
        y_train = y.astype(int)
        
        tempD = (y_pred-y_train)
        gradR = np.zeros(self.h.shape)
        gradR[self.h>0] = 1
        
        dWeights2 = np.dot(self.h.T, tempD)
        dBias2 = np.dot(np.ones((1,len(self.h))), tempD)
        dWeights1 = np.dot(x.T, gradR * np.dot(tempD, self.w2.T))
        dBias1 = np.dot(np.ones((len(x), 1)).T, gradR * np.dot(tempD, self.w2.T))
        
        self.w2, self.b2 = self.w2 - dWeights2 * lr / len(y_train), self.b2 - dBias2 * lr / len(y_train)
        self.w1, self.b1 = self.w1 - dWeights1 * lr / len(y_train), self.b1 - dBias1 *lr / len(y_train)
        pass