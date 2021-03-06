import numpy as np
import lin_reg as lr
import pylab as pl


def format_data(filename,dummy=False):
    """ Load in data and format it as basis for training or testing set """
    data = np.matrix(np.loadtxt(filename))
    Y = data[:,-1]
    X = data[:,0:-1]
    if dummy:
        X = np.concatenate((np.ones((len(X), 1)), X), 1)
    return(X,Y)

def add_rand_f(X,a):
    """ return input matrix with a random feature added to it """
    return np.concatenate((X, np.random.uniform(0, 10*a, (len(X), 1))), 1)

def add_empty_f(X):
    """ Return input matrix with features of 0 added to it """
    return np.concatenate((X, np.zeros((len(X), 1))), 1)

def main():

    #1 Load data
    (X_train,Y_train) = format_data("housing_train",dummy=True)
    (X_test, Y_test) = format_data("housing_test",dummy=True)

    #2 Compute Optimal Weights Vector
    reg_w_dum = lr.Lin_Reg(X_train,Y_train)
    print(reg_w_dum.get_w())

    #3 Compute Standard Square Error
    print(reg_w_dum.get_sse(Y_train,X_train))
    print(reg_w_dum.get_sse(Y_test,X_test))

    #4 Repeat SSE Computation without Dummy Variable
    (X_train_nod,Y_train_nod) = format_data("housing_train") # Notice no dummy selected
    (X_test_nod, Y_test_nod) = format_data("housing_test")
    reg_nod = lr.Lin_Reg(X_train_nod,Y_train_nod) # Form new linear regression
    print(reg_nod.get_sse(Y_train_nod,X_train_nod))
    print(reg_nod.get_sse(Y_test_nod,X_test_nod))

    #5 Add Random Features
    end_p=100 # end point of iteration
    X_train_randf = X_train # Supply training and testing sets w/ loaded data
    X_test_rf = X_test
    X_test_emp = X_test
    res_test_rf = np.zeros(end_p) # Pre-allocate space for results
    res_test_emp = np.zeros(end_p)
    res_train = np.zeros(end_p)

    for i in range(0,end_p): # Iteratively add features
        reg = lr.Lin_Reg(X_train_randf,Y_train) # Make new linear regression object
        res_train[i]=reg.get_sse(Y_train,X_train_randf) # Get results
        res_test_rf[i]=reg.get_sse(Y_test, X_test_rf)
        res_test_emp[i]=reg.get_sse(Y_test, X_test_emp)
        X_train_randf = add_rand_f(X_train_randf,i+1) # Add more features
        X_test_rf = add_rand_f(X_test_rf,i+1)
        X_test_emp=add_empty_f(X_test_emp)

    # Make plots
    pl.plot(range(0,end_p), res_train, 'r--', label='Base Features')
    pl.plot(range(0,end_p), res_test_emp, 'g^', label='With Empty Features')
    pl.plot(range(0,end_p), res_test_rf, 'bs', label = 'With Random Features')

    pl.title('Linear Regression with Various Numbers of Random Features')
    pl.ylabel('Sum of Squared Error (SSE)')
    pl.xlabel('Number of Random Features')
    pl.legend(loc='center left')

    pl.show()

    #Variant of Linear Regression in #6
    # Declare lambda values for regularization
    lambs=[0, 0.01, 0.05, 0.1, 0.5, 1, 5]
    res_train = []
    res_test = []

    for i in lambs: # Iterate thru the lambda values
        reg = lr.Lin_Reg(X_train,Y_train, lamb=i) # Create new object
        res_train.append(reg.get_sse(Y_train,X_train)) # Append result to list of results
        res_test.append(reg.get_sse(Y_test,X_test))

    # Plot all the results
    pl.plot(lambs, res_train)
    pl.title('Train Plot')
    pl.ylabel('train y')
    pl.xlabel('train x')
    pl.show()



    pl.plot(lambs, res_test)
    pl.title('Test Plot')
    pl.ylabel('test y')
    pl.xlabel('test x')
    pl.show()

main()