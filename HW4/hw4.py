import numpy as np
import matplotlib.pyplot as plt


import km



    


def main(): 
#1, implement K-means algorithm with k = 2, plot the objective (SSE) as a function of iterations.
#Present results of a typical run.

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


main();
#Each row in the data set is a 28x28 pixel representation of a digit
#30,000 rows


#pick k values randomly, 0-29999. for each value, find nearest k and store for that k
#Find the mean of these points, and the point closes to this mean. This is the new k location
#Iterate 



#2, Apply this implementation with different values of k {3..10} For each k, please run the alg. 10 times.

















