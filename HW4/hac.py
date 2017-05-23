
import numpy as np
import operator as op
import copy
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt


class HAC:
    def __init__(self, filename):
        self.data = np.genfromtxt(filename, delimiter=",")
        self.constructDistMat()
        #clusters exist as idxs for self.data
        self.clusters = [ [x] for x in   list(range(0, len(self.data)))]

    def constructDistMat(self):
        #memoize all possible N^2 connections in the complete graph
        self.distMat = np.zeros((len(self.data), len(self.data)))
        for x in range(0,len(self.data)):
            for y in range(x+1, len(self.data)): #don't do self
                self.distMat[x][y] = \
                        sum( [ (a - b)**2 for a,b in zip( self.data[x], self.data[y])])
                self.distMat[y][x] = self.distMat[x][y] #make life easier on me

#x and y are indexes in the self.cluster array
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

    def tab_merge(self, clst):
        # Add newly formed cluster to static list of clusters
        self.oclust.append(self.clusters[clst[0]] + self.clusters[clst[1]])
        nr = []
        nr.append(self.oclust.index(self.clusters[clst[0]]))
        nr.append(self.oclust.index(self.clusters[clst[1]]))
        if nr[0] > nr[1]:
           nr[0], nr[1] = nr[1], nr[0]
        nr.append(clst[2])
        nr.append(len(self.clusters[clst[0]]) + len(self.clusters[clst[1]]))
        self.ctab.append(nr)

    def link(self, k, method, dendo):
        if dendo:
            self.oclust = copy.deepcopy(self.clusters)
            self.ctab = []
        while len(self.clusters) > k:
            minGroups = (-1, -1, float('inf'))
            # mingroup is:
            # index of cluster a, index of cluster b, dist btw the two
            # my understanding of it
            # -Sam
            #find nearest clusters
            for x in range(0, len(self.clusters)):
                for y in range(x+1, len(self.clusters)):
                    tmp = self.linkDist(method, x, y)
                    if(tmp < minGroups[2]):
                        minGroups = (x, y, tmp)
            #merge clusters
            if dendo:
                self.tab_merge(minGroups)
                print(minGroups)
            self.clusters[minGroups[0]] += self.clusters[minGroups[1]]
            self.clusters.remove(self.clusters[minGroups[1]])
            #print(minGroups);
        return;


    def singleLink(self):
        self.link(10, 'single', False)
        return;

    def completeLink(self):
        self.link(10, 'complete', False)
        return;

    def draw_dendo(self):
        plt.figure()
        dn = hierarchy.dendrogram(self.ctab, color_threshold=0.)
        plt.show()



