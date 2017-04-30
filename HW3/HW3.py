import numpy as np
import KNN as knn
import DecTree as dt


def csv_to_array(filename):
    return np.genfromtxt(filename, delimiter=",")

def get_error(predictions, actual):
    incorrect = 0
    for i in range(len(predictions)):
        if predictions[i] != actual[i]:
            incorrect += 1
    return incorrect/len(predictions)

def main():

    # Load data
    test = csv_to_array("knn_test.csv")
    train = csv_to_array("knn_train.csv")

    # Part I

    # Set k
    k = 5

    knn_predictions = []
    myKNN = knn.KNN(train)
    for i in range(0, len(test)):
        knn_predictions.append(myKNN.predict(test[i], k))
    print("KNN predictions")
    print(knn_predictions)
    # Added in print error rate
    # Pretty high error rate, BUT < 0.5 so something is working
    # Pro'ly issue w/ algorithm itself, not our code
    print("Error rate: " + str(get_error(knn_predictions, test[:,0])))

    # KNN modifies the data,
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
