import numpy as np
import log_reg2 as lr

def format_data_csv(filename):
    """ Load in data and format it as basis for training or testing set """
    data = np.genfromtxt (filename, delimiter=",")
    Y = np.matrix(data[:,-1]).T
    X = np.matrix(data[:,0:-1])
    return(X,Y)


def format_data_csv2(filename):
    """ Load in data and format it as basis for training or testing set """
    data = np.genfromtxt (filename, delimiter=",")
    Y = data[:,-1]
    X = data[:,0:-1]
    return(X,Y)

def add_dummy(x):
    return np.concatenate((np.ones((len(x), 1)), x), 1)

def main():
    (X_train, Y_train) = format_data_csv2('../data/usps-4-9-train.csv')
    (X_test, Y_test) = format_data_csv2('../data/usps-4-9-test.csv')

    log_reg = lr.Log_Reg(add_dummy(X_train), Y_train, 'iteration', 0.1, 10)
#    print (log_reg.w)
#    print(log_reg.get_prob(X_test[300]))

if __name__ == "__main__":
    main()
