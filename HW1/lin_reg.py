import numpy as np

class Lin_Reg(object):
    def __init__(self, X, Y):
        # X is training sets features/attributes
        # Y is target ouptut
        # w is weights
        if (type(X) is not np.matrix) or (type(Y) is not np.matrix):
            raise ValueError("Data must be in form of numpy matrices")
        self.w=(X.T*X).I*(X.T*Y)

    def get_w(self):
        return self.w

    def get_sse(self,y,X):
        return (y - X*self.w).T * (y -X*self.w)
