import numpy as np
from scipy import stats

# Normalize ranges
def normalize_distances(train, test):
    for i in range(1, len(test)):
        span = np.amax(train[:,i]) - np.amin(train[:,i])
        for j in range(0, len(train[:,i])):
            train[j][i] = train[j][i] / span
        test[i] = test[i] / span

# Define distance function
def calculate_distance(a, b):
    assert(len(a) == len(b))
    total = 0
    for i in range(1, len(a)):
        total += (a[i] - b[i])*(a[i] - b[i])
    return total**0.5

# Get distances of all training data to test data
def get_distances(train, test):
    distances = []
    for i in range(0, len(train)):
        distances.append([calculate_distance(train[i], test), train[i][1]])
    return distances

# Currently takes one test sample, assumes the first column as label
class KNN(object):
    def __init__(self, train, test, k):
        normalize_distances(train, test)
        self.distances = get_distances(train, test)

        # Predict input
        self.distances[self.distances[:,0].argsort()]
        self.prediction = stats.mode(distances[0:k][1])