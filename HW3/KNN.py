import numpy as np

# Normalize ranges
def normalize_distances(train, test):
    for i in range(1, len(train[0])):
        feature_max = np.amax(train[:,i])
        for j in range(0, len(train[:,i])):
            train[j][i] = train[j][i] / feature_max
        for k in range(0, len(test[:,i])):
            test[k][i] = test[k][i] / feature_max
    return (train, test)

# Define distance function
def calculate_distance(a, b):
    assert(len(a) == len(b))
    total = 0
    for i in range(1, len(a)):
        total += (a[i] - b[i])*(a[i] - b[i])
    return total**0.5

# Get distances of all training data to test data
def get_distances(train, one_test_vector):
    distances = []
    for i in range(0, len(train)):
        single_distance = [calculate_distance(train[i], one_test_vector)]
        single_distance.append(train[i][0])
        distances.append(single_distance)
    return distances

# Currently takes one test sample, assumes the first column as label
class KNN(object):
    def __init__(self, train, test, k):
        self.predictions = []
        (train_n, test_n) = normalize_distances(train, test)
        for i in range(0, len(test)):
            self.unsorted_distances = get_distances(train_n, test_n[i])

            # Top K nearest distances with predictions
            self.sorted_distances = sorted(self.unsorted_distances, key=lambda tup: tup[0])[0:k]

            # Alternative quick mode, specific to labels -1 and 1 (stats.mode would operate better on differently structured arrays)
            my_sum = 0
            for j in range(0, k):
                my_sum += self.sorted_distances[j][1]
            if(my_sum >= 0):
                self.predictions.append(1)
            elif(my_sum , 0):
                self.predictions.append(-1)
"""
def main():
    a = [[-1., 1., 2., 3.],[-1., 4., 5., 6.],[1., 7., 8., 9.]]
    b = [1., 3., 5., 5.]
    train = np.asarray(a)
    test = np.asarray(b)
    myKNN = KNN(train, test, 1)


main()

# -1 1 2 3
# -1 4 5 6
#  1 7 8 9

# 1 3 5 5
"""