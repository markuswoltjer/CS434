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

def add_rand_f(X,a):
    return np.concatenate((X, np.random.uniform(0, 10*a, (len(X), 1))), 1)

def add_empty_f(X):
    return np.concatenate((X, np.zeros((len(X), 1))), 1)

def main():

    #1 Load data
    (X_train,Y_train) = format_data("housing_train",dummy=True)
    (X_test, Y_test) = format_data("housing_test",dummy=True)
    #2 compute optimal weights
    reg_w_dum = lr.Lin_Reg(X_train,Y_train)
    print(reg_w_dum.get_w())
    #3 compute sse
    print(reg_w_dum.get_sse(Y_train,X_train))
    print(reg_w_dum.get_sse(Y_test,X_test))
    #4 Repeat w/o dummy
    (X_train_nod,Y_train_nod) = format_data("housing_train")
    (X_test_nod, Y_test_nod) = format_data("housing_test")
    reg_nod = lr.Lin_Reg(X_train_nod,Y_train_nod)
    print(reg_nod.get_sse(Y_train_nod,X_train_nod))
    print(reg_nod.get_sse(Y_test_nod,X_test_nod))
    #5 Throw in random features
    end_p=100
    X_train_randf = X_train
    X_test_rf = X_test
    X_test_emp = X_test
    res_test_rf = np.zeros(end_p)
    res_test_emp = np.zeros(end_p)
    res_train = np.zeros(end_p)
    for i in range(0,end_p):
        reg = lr.Lin_Reg(X_train_randf,Y_train)
        res_train[i]=reg.get_sse(Y_train,X_train_randf)
        res_test_rf[i]=reg.get_sse(Y_test, X_test_rf)
        res_test_emp[i]=reg.get_sse(Y_test, X_test_emp)
        X_train_randf = add_rand_f(X_train_randf,i+1)
        X_test_rf = add_rand_f(X_test_rf,i+1)
        X_test_emp=add_empty_f(X_test_emp)

    plt.plot(range(0,end_p), res_train, 'r--', range(0,end_p), res_test_rf, 'bs', range(0,end_p), res_test_emp, 'g^')
    plt.show()
    #6
    lambs=[0, 0.01, 0.05, 0.1, 0.5, 1, 5]
    res_train = []
    res_test = []
    for i in lambs:
        reg = lr.Lin_Reg(X_train,Y_train, lamb=i)
        res_train.append(reg.get_sse(Y_train,X_train))
        res_test.append(reg.get_sse(Y_test,X_test))

    plt.plot(lambs, res_train)
    plt.show()
    plt.plot(lambs, res_test)
    plt.show()

main()