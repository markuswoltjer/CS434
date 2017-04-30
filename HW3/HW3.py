import numpy as np
from KNN import KNN as knn
import DecTree as dt


def csv_to_array(filename):
    return np.genfromtxt(filename, delimiter=",")

def main():
    # Load data
    test = csv_to_array("knn_test.csv")
    train = csv_to_array("knn_train.csv")


    # Set k

    k = 5

    # predictions = []
    # for i in range(0, len(test)):
    #     myKNN = knn(train, test[i], k)
    #     predictions.append(myKNN.prediction)
    #
    # print(predictions)

    # Part II
    print("Part II\n")
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
