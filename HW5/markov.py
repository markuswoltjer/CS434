import random
import copy

#actions is an action x P(states) 2-d Array
class Markov:
    def __init__(self, states, reward ):
        self.states = states
        self.reward = reward
        #actions is a action by stateK by stateL array of probabilities from K to L

    #Discover Utility for each state
    def Bellman(self, discFactor, deltStop):
        
        utility = copy.deepcopy(self.reward) 
        uTmp = copy.deepcopy(utility)

        maxDelt = 1.0#do part for the do while
 
        while maxDelt > deltStop:
            maxDelt = 0
            for i in range(0, len(self.states)):
                most = max([sum([ x*y for x,y in zip(utility,act)]) for act in self.states[i]])
                uTmp[i] = self.reward[i] + (most * discFactor)
                
                delta = uTmp[i] - utility[i]
                if abs(delta) > maxDelt:
                    maxDelt = abs(delta)
            utility = copy.deepcopy(uTmp)




