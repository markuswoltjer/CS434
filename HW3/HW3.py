import numpy as np
import KNN as knn
import DecTree as dt


def csv_to_array(filename):
    return np.genfromtxt(filename, delimiter=",")

def main():
    # Load data
    test = csv_to_array("knn_test.csv")
    train = csv_to_array("knn_train.csv")

    # Set k
    k = 5

    myKNN = knn.KNN(train, test, k)
    print(myKNN.predictions)

main()
