import numpy as np

def main():

    #load training data
    train_data = np.matrix(np.loadtxt("housing_train"))

    #load testing data
    test_data = np.matrix(np.loadtxt("housing_test"))

    #split training data into attributes and labels
    Y = train_data[:,-1]
    X_no_dummy = train_data[:,0:-1]

    #prepend dummy column for "b multipliers"
    dummy = np.ones((len(X_no_dummy), 1))
    X = np.concatenate((dummy, X_no_dummy), 1)

    #compute optimal weights
    w = (X.T*X).I*(X.T*Y)


    print(X)

main()