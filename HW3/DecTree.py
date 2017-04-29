import numpy as np
import math
import numbers


def get_entropy(data):
    p_pos = (data[:,0] == 1).sum() / len(data)
    p_neg = (data[:,0] == -1).sum() / len(data)
    if p_pos == 0 or p_neg == 0:
        return 0
    else:
        return - p_pos * math.log2(p_pos) \
           - p_neg * math.log2(p_neg)


def get_info_gain(thresh, data):
    data_left = data[data[:,1] < thresh]
    data_right = data[data[:,1] >= thresh]
    p_left = len(data_left) / len(data)
    p_right = len(data_right) / len(data)
    return get_entropy(data) \
            - p_left * get_entropy(data_left) \
            - p_right * get_entropy(data_right)

def get_thresh(data):
    # data's 0th column is output Y
    # data's 1st column is feature x_i
    # Sort data by feature column
    data_sort = data[data[:, 1].argsort()]
    best_gain = - float("inf")
    best_thresh = np.median(data, axis=0)[1]
    # get a list of all possible thresholds
    threshs = [(data_sort[x, 1] + data_sort[x + 1, 1]) / 2\
                for x in range(len(data_sort) - 1)]
    for thresh in threshs:
        gain = get_info_gain(thresh, data)
        if gain >= best_gain:
            best_gain = gain
            best_thresh =thresh

    return best_thresh, best_gain


def get_test_vals(data):
    best_gain = - float("inf")
    for feature in range(1, len(data[0])):
        # iterate through each feature
        # Pass get thresh just two columns, output and selected feature
        thresh, gain = get_thresh(np.compress(
                            [x is 0 or x is feature for x in range(len(data[0]))],
                            data, axis=1))
        if gain > best_gain:
            best_thresh = thresh
            best_gain = gain
            best_feat = feature

    return best_thresh, best_feat


class DecTest(object):
    def __init__(self, data):
        self.thresh, self.feat = get_test_vals(data)

    def data_split(self, data):
        return data[data[:, self.feat] < self.thresh], data[data[:, self.feat] >= self.thresh]

    def get_direction(self, case):
        if case[self.feat] < self.thresh:
            return "left"
        else:
            return "right"

class DecTree(object):
    def __init__(self, data, depth):
        # Should take in data (correctly formatted area)
        # Design a test for that data
        #       i.e. determine feature w/ threshold that offers
        #       maximum info-gain
        # then assign the two leaves
        # either with predictions or new trees
        self.test = DecTest(data)
        data_left, data_right = self.test.data_split(data)
        if depth is 1:
            self.left = (data_left[:,0] == 1).sum() / len(data_left)
            self.right = (data_right[:,0] == 1).sum() / len(data_right)
        else:
            if data_left.size == len(data[0]):
                self.left = 1. if data_left[0,0] == 1 else 0.
            else:
                self.left = DecTree(data_left, depth - 1)
            if data_right.size == len(data[0]):
                self.right = 1. if data_right[0,0] == 1 else 0.
            else:
                self.right = DecTree(data_right, depth - 1)

    def search(self, case):
        if self.test.get_direction(case) == "left":
            if isinstance(self.left, numbers.Real):
                return self.left
            else:
                return self.left.search(case)
        elif self.test.get_direction(case) == "right":
            if isinstance(self.right, numbers.Real):
                return self.right
            else:
                return self.right.search(case)
