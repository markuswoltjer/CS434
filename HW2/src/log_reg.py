import numpy as np

# Helper fxns


def g(w, x_i):
    # assumes x and w have same dimensions going in
    return (1 + np.exp(-1 * (w.T * x_i)))**-1
    #w_t = w.T
    #to_inv = (w * x_i.T)


def l(g_w, w, x, y):
    return -y * np.log(g_w(w, x.T)) - (1 - y) * np.log(1 - g_w(w, x.T))


def L(w, X, Y):
    L = 0
    for i in range(len(X)):
        L += l(g, w, X[i], Y[i])
    return L


class Log_Reg(object):
    def __init__(self, X, Y, con_key, lr, thresh):
        self.thresh = thresh
        self.iter = 0 # no of iterations
        self.norm = float("inf")
        self.obj = float("inf")
        self.obj_ch = float("inf")
        self.w = np.matrix(np.zeros(np.shape(X)[1])).T
        while not self.converge(con_key):
            d = np.matrix(np.zeros(np.shape(X)[1]))
            for i in range(len(Y)):
                y_hat_i = g(self.w, X[i].T)
                y_i = Y[i]
                error = (Y[i] - y_hat_i)[0,0]
                add_to_d = error * X[i]
                d += add_to_d
            self.w += lr * d.T
            #Update parameters for converge
            self.iter += 1
            self.obj_prev = self.obj
            self.obj=L(self.w, X, Y)
            self.obj_ch = abs(self.obj - self.obj_prev)
            self.norm = np.linalg.norm(d)

    def converge(self, key):
        return{
            'objective': self.obj_ch <= self.thresh,
            'gradient' : self.norm <= self.thresh,
            'iteration': self.iter >= self.thresh
        }[key]

    def get_prob(self,x_i):
        return g(self.w, x_i.T)