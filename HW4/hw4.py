###############################
#CS434: Erich Kramer, Sam Jacobs, Markus Woltjer
#To run: use python3 in environ. which has numpy and matplot lib
###############################

import numpy as np
import matplotlib.pyplot as plt

import km
import hac

def partOne():
    kTwo = km.kmean("data-2.txt", 2)

    SSETrend = kTwo.iterSolve()
    
    print("Trend is: ", SSETrend)

    plt.plot(SSETrend, 'b--')
    plt.ylabel("SSE values")
    plt.xlabel("K-means iteration")
    plt.show()


def partTwo():
    kOBJ = km.kmean("data-2.txt", 0)
    bestSSE = np.full(11, np.inf);

    print("Solving kmean 10 iteration on 150 data points")
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
    hacker = hac.HAC("data-2.txt")
    hacker.singleLink()
    hacker.link(1,'single', True)
    hacker.draw_dendo()


#4  Full linked HAC
    hacker = hac.HAC("data-2.txt")
    hacker.completeLink()
    hacker.link(1,'complete', True)
    hacker.draw_dendo()


main();


















