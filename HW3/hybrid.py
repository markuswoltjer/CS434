import numpy as np
import KNN as knn
import DecTree as dt

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
def get_distances_with_indices(train_n, one_test_vector):
    distances = []
    for i in range(0, len(train_n)):
        single_distance = [calculate_distance(train_n[i][1:], one_test_vector)]
        single_distance.append(i)
        distances.append(single_distance)
    return distances

# Currently takes one test sample, assumes the first column as label
class hybrid(object):
    def __init__(self, train, input_vector, k, depth):
        (self.train_n, self.scalars) = normalize_distances(train)
        self.scaled_input = np.empty(len(input_vector)-1)
        for i in range(0, len(self.scalars)):
            self.scaled_input[i] = input_vector[i+1] / self.scalars[i]
        self.unsorted_distances = get_distances_with_indices(self.train_n, self.scaled_input)
        # Top K nearest distances with predictions
        self.sorted_distances = sorted(self.unsorted_distances, key=lambda tup: tup[0])[0:k]
        self.candidates = []
        for j in range(0, len(self.sorted_distances)):
            self.candidates.append(train[self.sorted_distances[j][1]]) #add near points from the training data to another array
        self.tree = dt.DecTree(self.candidates, depth)
        self.tree.print_tree()

    def get_error(self,input_vector):
        self.tree.get_error(input_vector)

# How might this be extended for multiple inputs? Our suggestion is to find the n-dimensional centroid of the inputs,
# and use that as the "input_vector" to find relevant training points, and then passing all of the inputs into the tree from that