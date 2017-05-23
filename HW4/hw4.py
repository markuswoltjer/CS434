import numpy as np
import matplotlib.pyplot as plt

import km
import hac

def partOne():
    print("Generating object")
    kTwo = km.kmean("data.txt", 2)

    print("Solving kmean")
    SSETrend = kTwo.iterSolve()
    
    print("Trend is: ", SSETrend)
    print("Size is: ", len(SSETrend))


    plt.plot(SSETrend, 'b--')
    plt.ylabel("SSE values")
    plt.xlabel("K-means iteration")
    plt.show()


def partTwo():
    kOBJ = km.kmean("data-2.txt", 0)
    bestSSE = np.full(11, np.inf);

    for k in range(3, 11):
        print("10 iters of K value: ", k)
        for i in range(0, 10):
            kOBJ.reset(k)
            kOBJ.iterSolve()
            if( kOBJ.SSE < bestSSE[k]):
                bestSSE[k] = kOBJ.SSE
                
    plt.plot(bestSSE, 'b--')
    plt.ylabel("Best SSE over 10 iter")
    plt.xlabel("k value")
    plt.show()


    


def main(): 
#1, implement K-means algorithm with k = 2, plot the objective (SSE) as a function of iterations.
#Present results of a typical run.

    partOne()

#2, Apply this implementation with different values of k {3..10} For each k, please run the alg. 10 times.

    partTwo()

#3 Single linked HAC

#    hacker = hac.HAC("data-2.txt")
#    hacker.completeLink()
#    for x in hacker.clusters:
#        print(x)
#    hacker.link(1,'complete', False)

#4  Full linked HAC

main();
#Each row in the data set is a 28x28 pixel representation of a digit
#30,000 rows


#pick k values randomly, 0-29999. for each value, find nearest k and store for that k
#Find the mean of these points, and the point closes to this mean. This is the new k location
#Iterate 


















