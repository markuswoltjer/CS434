import numpy as numpy
from scipy import stats

class KNN(object):
    def __init__(train, test, k):
        normalize_distances(train, test)
        self.distances = get_distances(train, test)

        # Predict input
        self.distances[self.distances[:,0].argsort()]
        self.prediction = stats.mode(distances[0:k][1])

    # Normalize ranges
    def normalize_distances(train, test):
        for i in range(0, len(test)):
            span = np.amax(train[:,i]) - np.amin(train[:,i])
            for j in range(0, len(train[:,i])):
                train[j][i] = train[j][i] / span
            test[i] = test[i] / span

    # Define distance function
    def calculate_distance(a, b):
        assert(len(a) == len(b))
        total = 0
        for i in range(0, len(a)):
            total += (a[i] - b[i])*(a[i] - b[i])
        return sqrt(total)

    # Get distances of all training data to test data
    def get_distances(train, test):
        distances = []
        for i in range(0, len(train)):
            distances.append([calculate_distance(train[i], test), train[i][-1]])
        return distances