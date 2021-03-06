import numpy as np

# Normalize ranges
def normalize_distances(train):
    scalars = []
    train_n = np.copy(train)
    for i in range(1, len(train[0])):
        feature_max = np.amax(train[:,i])
        for j in range(0, len(train[:,i])):
            train_n[j][i] = train[j][i] / feature_max
        scalars.append(feature_max)
    return (train_n, scalars)

# Define distance function
def calculate_distance(a, b):#only features passed in here
    total = 0
    for i in range(0, len(a)):
        total += (a[i] - b[i])*(a[i] - b[i])
    return total**0.5

# Get distances of all training data to test data
def get_distances(train_n, one_test_vector):
    distances = []
    for i in range(0, len(train_n)):
        single_distance = [calculate_distance(train_n[i][1:], one_test_vector)]
        single_distance.append(train_n[i][0])
        distances.append(single_distance)
    return distances

# Currently takes one test sample, assumes the first column as label
class KNN(object):
    def __init__(self, train):
        (self.train_n, self.scalars) = normalize_distances(train)

    def predict(self, input_vector, k):
        self.scaled_input = np.empty(len(input_vector)-1)
        for i in range(0, len(self.scalars)):
            self.scaled_input[i] = input_vector[i+1] / self.scalars[i]
        self.unsorted_distances = get_distances(self.train_n, self.scaled_input)
        # Top K nearest distances with predictions
        self.sorted_distances = sorted(self.unsorted_distances, key=lambda tup: tup[0])[0:k]
        # Alternative quick mode, specific to labels -1 and 1 (stats.mode would operate better on differently structured arrays)
        my_sum = 0
        for j in range(0, k):
            my_sum += self.sorted_distances[j][1]
        if(my_sum >= 0):
            self.prediction = 1
        elif(my_sum < 0):
            self.prediction = -1
        return self.prediction