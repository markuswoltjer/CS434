import numpy as np
import lin_reg as lr
import matplotlib.pyplot as plt


def format_data(filename,dummy=False):
    data = np.matrix(np.loadtxt(filename))
    Y = data[:,-1]
    X = data[:,0:-1]
    if dummy:
        X = np.concatenate((np.ones((len(X), 1)), X), 1)
    return(X,Y)

def main():

    (X_train,Y_train) = format_data("housing_train",dummy=True)
    (X_test, Y_test) = format_data("housing_test",dummy=True)
    #1 compute optimal weights
    reg_w_dum = lr.Lin_Reg(X_train,Y_train)
    print(reg_w_dum.get_w())
    #2 compute sse
    print(reg_w_dum.get_sse(Y_train,X_train))
    print(reg_w_dum.get_sse(Y_test,X_test))
    #3 Repeat w/o dummy
    (X_train_nod,Y_train_nod) = format_data("housing_train")
    (X_test_nod, Y_test_nod) = format_data("housing_test")
    reg_nod = lr.Lin_Reg(X_train_nod,Y_train_nod)
    print(reg_nod.get_sse(Y_train_nod,X_train_nod))
    print(reg_nod.get_sse(Y_test_nod,X_test_nod))




main()