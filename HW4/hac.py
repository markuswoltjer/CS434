
import numpy as np



class HAC:
    def __init__(self, filename):
        self.data = np.genfrontxt(filename, delimiter",")

        self.constructDistMat()
        

    def constructDistMat(self):

        self.distMat = np.zeros((len(self.data), len(self.data)))
        for x in range(0,len(self.data)):
            

