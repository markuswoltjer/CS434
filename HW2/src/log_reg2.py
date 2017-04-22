import numpy as np
import math


def sigmoid(w, x):
    p = 1/(1 + np.exp(-np.dot(w, x)))
    # np.dot should return the same result as w.T * x, but can work properly on arrays
    return p


def l(y, x, w):
    if y == 1:
        return -np.log(sigmoid(w, x))
    else:
        return -np.log(1 - sigmoid(w, x))


def L(Y, X, w):
    to_return = 0
    for i in range(len(Y)):
        to_return += l(Y[i], X[i], w)
    return to_return


def predict_accuracy(w, X, Y):
    correct = 0
    for i in range(len(Y)):
        prediction = sigmoid(w,X[i])
        if ((prediction > 0.5) and (Y[i]==1)) or ((prediction < 0.5) and (Y[i]==0)):
            correct += 1
    return correct/len(Y)

class Log_Reg(object):
    def __init__(self, X, Y, con_key, lr, thresh, lamb=0):
        self.w = np.zeros(len(X[0]))
        self.thresh = thresh
        self.iter = 0
        self.norm = float("inf")
        self.obj = float("inf")
        self.obj_ch = float("inf")
        while not self.converge(con_key):
            d = np.zeros(len(X[0]))
            for i in range(len(Y)):
                y_hat_i = sigmoid(self.w, X[i])
                error = Y[i] - y_hat_i
                d = d + error * X[i]
            self.w += lr * d - lr * lamb * self.w
            self.iter += 1
            self.obj_prev = self.obj
            self.obj = L(Y, X, self.w)
#            print(self.obj)
            self.accuracy = predict_accuracy(self.w, X, Y)
            self.obj_ch = abs(self.obj - self.obj_prev)
            self.norm = np.linalg.norm(-d)

    #        print('done')

    def converge(self, key):
        return{
            'objective': self.obj_ch <= self.thresh,
            'gradient': self.norm <= self.thresh,
            'iteration': self.iter >= self.thresh
        }[key]


    def predict(self,x):
        return sigmoid(self.w,x)


    def another_batch(self, X, Y, lr):
        d = np.zeros(len(X[0]))
        for i in range(len(Y)):
            y_hat_i = sigmoid(self.w, X[i])
            error = Y[i] - y_hat_i
            d = d + error * X[i]
        self.w += lr * d
        self.iter += 1
        self.obj_prev = self.obj
        self.obj = L(Y, X, self.w)
        #            print(self.obj)
        self.accuracy = predict_accuracy(self.w, X, Y)
        self.obj_ch = abs(self.obj - self.obj_prev)
        self.norm = np.linalg.norm(d)
