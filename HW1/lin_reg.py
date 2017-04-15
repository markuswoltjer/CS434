import numpy as np

class Lin_Reg(object):
    def __init__(self, X, Y,lamb=0):
        # X is training sets features/attributes
        # Y is target ouptut
        # w is weights

        if (type(X) is not np.matrix) or (type(Y) is not np.matrix):
            raise ValueError("Data must be in form of numpy matrices")

        self.w=np.matrix((X.T*X + np.matrix(lamb*np.identity(len(X.T*X))))).I*(X.T*Y)

    def get_w(self):
        """ Get weights used by linear regression object"""
        return self.w

    def get_sse(self,y,X):
        """ Pass in target outputs and input features
            Get Sum Square Error"""
        return ((y - X*self.w).T * (y - X*self.w)).item(0)


