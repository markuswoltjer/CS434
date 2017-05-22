import numpy as np
import random
'''
    data file is a 764 feature set of 6000 values
    this is the updated, previously as 30,000
'''

class kmean:
    def __init__(self, filename, k):
        self.data = np.genfromtxt(filename, delimiter=",")
        self.k = k
        #get k unique indexes in data and store them
        self.means = [self.data[x] for x in random.sample(range(0, len(self.data)), k) ]

        self.iterations = 0
        self.SSE = 0

    #changes state, no return value
    def iterate(self):
        #new means, init to 0
        nextMean = np.zeros((len(self.means), 784))
        nextSize = np.zeros(len(self.means))
        self.SSE = 0

        #for each entry
        for entry in range(0, len(self.data)):

            #tuple of mean index and error value
            best = (-1, float("inf"))

            #for each mean
            for mean in range(0, self.k):
                error = sum( [ (x - y)**2 for x,y in zip( self.means[mean], self.data[entry] ) ] )
                dist = error**.5
                if(dist < best[1]):
                    best = (mean, dist)

            #factor this entry into it's respective mean
            nextMean[best[0]] = [x + y for x,y in zip( nextMean[best[0]],  self.data[entry]) ]
            #count elements for average to assign next mean
            nextSize[best[0]] += 1

            self.SSE += error

        #reassign means using aggregate new means and count of points
        self.means = [ [ x / y for x in m] for m,y in zip(nextMean, nextSize)]
        self.iterations += 1
        return 0

    def iterSolve(self):
        iterSSE = []
        prev = -1
        curr = 0

        while prev != curr:
            self.iterate()
            iterSSE += [self.SSE]
            prev = curr
            curr = self.SSE

        return iterSSE

    #this way we don't have to reload data from txt file
    def reset(self, newK):
        self.k = newK
        self.means = [self.data[x] for x in random.sample(range(0, len(self.data)), newK) ]
        self.iterations = 0
