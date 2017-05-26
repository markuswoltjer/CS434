# Use instructions:
# All the submitted files should be housed in the same directory
# Part two requires python 3
# The standard scipy, numpy, and matplotlib libraries are needed, which in linux can be installed with sudo apt-get install python-numpy python-scipy python-matplotlib
# Some of our teammates had trouble with the pylab library in python3. The workaround is to first run with python 2 to get the plot for the KNN, and then run with python3
# the decision tree to work.
# If any of the above are giving you trouble, please email me (Markus), since I check my email frequently.

import numpy as np
import KNN as knn
import DecTree as dt
#import pylab as pl
import hybrid as hb

def csv_to_array(filename):
    return np.genfromtxt(filename, delimiter=",")

def get_error(predictions, actual):
    incorrect = 0
    for i in range(len(predictions)):
        if predictions[i] != actual[i]:
            incorrect += 1.
    return incorrect/float(len(predictions))

def get_training_error(train, k):
    myKNN = knn.KNN(train)
    num_errors = 0
    for i in range(0, len(train)):
        if(myKNN.predict(train[i], k) != train[i][0]):
            num_errors += 1
    return num_errors

def get_leave_one_out_error(train, k):
    num_errors = 0
    for i in range(0, len(train)):
        train_minus = np.delete(train, i, axis=0)
        myKNN = knn.KNN(train_minus)
        if(myKNN.predict(train[i], k) != train[i][0]):
            num_errors += 1
    return num_errors

def get_test_error(train, test, k):
    myKNN = knn.KNN(train)
    num_errors = 0
    for i in range(0, len(test)):
        if(myKNN.predict(test[i], k) != test[i][0]):
            num_errors += 1
    return num_errors

def main():
    # test VCS commit
    # Load data
    test = csv_to_array("knn_test.csv")
    train = csv_to_array("knn_train.csv")

    # Part I

    # Set k (5 chosen just for demonstration, feel free to modify)
    k = 5
    
    knn_predictions = []
    myKNN = knn.KNN(train)
    for i in range(0, len(test)):
        knn_predictions.append(myKNN.predict(test[i], k))
    print("KNN predictions")
    print(knn_predictions)
    print("test labels")
    print(test[:,0])
    print("Error rate: " + str(get_error(knn_predictions, test[:,0])))

    training_error = []
    leave_one_out_error = []
    test_error = []

    for k in range(1, 52, 2):
        # Training error as number of mistakes
        training_error.append(get_training_error(train, k))
        # Leave-one-out training error
        leave_one_out_error.append(get_leave_one_out_error(train, k))
        # Test error
        test_error.append(get_test_error(train, test, k))

    # Outputs
    print("training error")
    print(training_error)

    print("leave one out error")
    print(leave_one_out_error)

    print("test error")
    print(test_error)

    # Re-load data for the decision tree
    test = csv_to_array("knn_test.csv")
    train = csv_to_array("knn_train.csv")

    # Part II
    print("Part II\n")
    print("NB: use python3 interpreter to work properly")
    # Problem 1
    stump = dt.DecTree(train, 1)
    print("Problem 1: \n")
    stump.print_tree()
    print("\nError rate for training data = " + str(stump.get_error(train)))
    print("Error rate for test data = " + str(stump.get_error(test)) + "\n")
    # Problem 2
    tree = dt.DecTree(train, 6)
    print("Problem 2:")
    print("(NB: Some paths exhaust data splits before final depth)\n")
    tree.print_tree()
    print("\nError rate for training data = " + str(tree.get_error(train)))
    print("Error rate for test data = " + str(tree.get_error(test)) + "\n")

    # Extra Credit
    # The plan is similar to SVM, which draws a barrier between classes
    # based on the relevant (near) points to the barrier. In this case,
    # we are going to only consider the K nearest points to a test input
    # and instead of just having them vote, they will be used as the
    # training set for a decision tree.

    #myHybrid = hb.hybrid(train, test[0], 30, 3)
    #print(myHybrid.get_error(test[0]))

    #The EC implementation is a little buggy but the idea is there, and it's logically laid out in hybrid.py



main()
