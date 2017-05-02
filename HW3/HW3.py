import numpy as np
import KNN as knn
import DecTree as dt
import pylab as pl

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

    #Set k
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

    # Plot
    pl.plot(range(1, 52, 2), training_error, 'r--', label='Train Error')
    pl.plot(range(1, 52, 2), leave_one_out_error, 'g--', label='Leave-One-Out Error')
    pl.plot(range(1, 52, 2), test_error, 'b--', label = 'Test Error')

    pl.title('KNN Performance Dependent on K')
    pl.ylabel('K')
    pl.xlabel('Total Incorrect Predictions')
    pl.legend(loc='upper left')

    pl.show()

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

main()
