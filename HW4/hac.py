
import numpy as np
import operator as op

class HAC:
    def __init__(self, filename):
        self.data = np.genfromtxt(filename, delimiter=",")
        self.constructDistMat()
        #clusters exist as idxs for self.data
        self.clusters = [ [x] for x in   list(range(0, len(self.data)))]
#        print(self.clusters)

    def constructDistMat(self):
        #memoize all possible N^2 connections in the complete graph
        self.distMat = np.zeros((len(self.data), len(self.data)))
        for x in range(0,len(self.data)):
            for y in range(x+1, len(self.data)): #don't do self
                self.distMat[x][y] = \
                        sum( [ (a - b)**2 for a,b in zip( self.data[x], self.data[y])])
                self.distMat[y][x] = self.distMat[x][y] #make life easier on me

#x and y are indexes in the self.cluster array
    def singleLinkDist(self, x, y):
        #closest = (-1, -1, float('inf'))
        closest = float('inf')
        for a in self.clusters[x]:
            for b in self.clusters[y]:
                if self.distMat[a][b] < closest:
                    #closest = (a, b, self.distMat[a][b])
                    closest = self.distMat[a][b]

        return closest

    def completeLinkDist(self, x, y):
        completeDist = 0
        for a in self.clusters[x]:
            for b in self.clusters[y]:
                completeDist += self.distMat[a][b]

        return completeDist

    def linkDist(self, method, x, y):
        ops = {'single': op.lt,
               'complete': op.gt}
        refs = {'single': float('inf'),
                'complete': 0}
        dist = refs[method]
        for a in self.clusters[x]:
            for b in self.clusters[y]:
                if ops[method](self.distMat[a][b], dist):
                    dist = self.distMat[a][b]

        return dist


    def link(self, k, method):
        while len(self.clusters) > k:
            minGroups = (-1, -1, float('inf'))

            #find nearest clusters
            for x in range(0, len(self.clusters)):
                for y in range(x+1, len(self.clusters)):
                    tmp = self.linkDist(method, x, y)
                    if(tmp < minGroups[2]):
                        minGroups = (x, y, tmp)

            #merge clusters
            self.clusters[minGroups[0]] += self.clusters[minGroups[1]]
            self.clusters.remove(self.clusters[minGroups[1]])
            #print(minGroups);

        return;


    def singleLink(self):
        self.link(10, 'single')
        return;

    def completeLink(self):
        self.link(10, 'complete')
        return;



