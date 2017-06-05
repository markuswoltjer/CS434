import random


#actions is an action x P(states) 2-d Array
class state:
    def __init__(self, actions, reward, idx ):
        self.markov = actions
        self.reward = reward
        self.idx    = idx
    #does not require current state, this is a state object.

    def transition(self, action):
        prob = random.random() #[0.0, 1.0)
        #get distribution for this action for this state
        distribution = self.markov[action] 
    
        idx = 0
        while prob > 0:
            prob -= distribution[idx]
            idx += 1
        return idx

    def display(self):
        print("State ", self.idx)
        for x in self.markov:
            print(x)
        print("Reward is: ", self.reward)
            

