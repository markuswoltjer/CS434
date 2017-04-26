import numpy as numpy
import KNN as knn
def csv_to_array(filename):
    return = np.genfromtxt(filename, delimiter=",")

def main():
    # Load data
    test = csv_to_array("knn_test.csv")
    train = csv_to_array("knn_train.csv")

    # Set k
    k = 5

    predictions = []
    for i in range(0, len(test)):
        myKNN = knn(train, test[i], k)
        predictions.append(myKNN.prediction)

    print(predictions)
    
main()
