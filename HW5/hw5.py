#!/bin/python3
# CS434 - HW5 
# Sam Jacobs, Erich Kramer, Markus Woltjer
#
#
import sys
import os.path
import numpy as np
from markov import *
# Build a planner. Value iteration algorithm

#input - MDP and factor discount B, output optimal utility function and policyP for mdp 
#and discount factor


#2 Test on provided data with different B values

#ouptut should consist of B, nx1 U and nx1 P


def cmdArgs():
    if len(sys.argv) != 3 :
        print("Usage: ./hw5 [filename] [discount factor [0,1]]")
        exit()    
    elif not os.path.exists(sys.argv[1]):
        print("Bad filename!")
        exit()
    else:
        try:
            x = float(sys.argv[2])
            if( x < 0 or x > 1):
                raise ValueError
        except ValueError as error:
            print("Bad Discount factor! Must be float between 0 and 1")

            exit()
        return;

#assumes input file is filled with action by state tables for each state
#readlines probably better than readline() for refactor
def getMark(numStates, numActions, fd):
    
    tmp = []
    for x in range(0, numActions):
        tmp.append([])
        fd.readline()
        for y in range(0, numStates):
            tmp[x].append([ float(val) for val in fd.readline().rstrip('\n').rsplit('   ')])
            #Note: THIS ACCEPTS ONLY *THREE SPACES* DELIMITER. 
    #list is currently [action0 [state0 ... stateN] , ... , actionN [state0 ... stateN] ]
    fd.readline()
    rewards = [float(x) for x in fd.readline().rstrip('\n').rsplit('   ')]
    #swizzel
    states = []
    for i in range(0, numStates):
        stateProb = [ x[i] for x in tmp ]
        states.append( stateProb)


    m = Markov(states, rewards)
    return m

#unicode characters <3
def δ(beta):
    epsilon = 10**-10
    numerator = epsilon*((1-beta)**2)
    denominator = 2 * (beta**2)
    return numerator / denominator



def main():
    cmdArgs()
    β = float(sys.argv[2])
    delta = δ(β)

    f = open(sys.argv[1], 'r')
    (statCnt, actCnt) = [ int(x) for x in f.readline().rstrip('\n').rsplit(' ') ]

    markov = getMark(statCnt,actCnt, f)
    markov.Bellman( β, delta)
    markov.displayOutput()

    f.close()
    return;



main()

