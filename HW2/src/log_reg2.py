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


class Log_Reg(object):
    def __init__(self, X, Y, con_key, lr, thresh):
        self.w = np.zeros(len(X[0]))
        self.thresh = thresh
        self.iter = 0
        while not self.converge(con_key):
            d = np.zeros(len(X[0]))
            for i in range(len(Y)):
                y_hat_i = sigmoid(self.w, X[i])
                error = Y[i] - y_hat_i
                d = d + error * X[i]
            self.w += lr * d
            self.iter += 1
#        print('done')

    def converge(self, key):
        return{
            'iteration': self.iter >= self.thresh
        }[key]

    def predict(self,x):
        return sigmoid(self.w,x)