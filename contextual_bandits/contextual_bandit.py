
import numpy as np

"""
    let's play with 4 bandits and 4 arms
    minimum in the array is the optimal actions,
    we will validate our agent to be able to learn these best action
"""
class contextual_bandit:
    def __init__(self):
        self.bandits = np.array([[7,9,-5,10],
                                 [2,-5,11,5],
                                 [-5,9,11,15],
                                 [15,-5,11,0]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]
        self.state = 0
    
    def start(self):
        """
            start our bandit game
            return:
                state - randomly choose a state
        """     
        self.state = np.random.randint(0, self.num_bandits)
        return self.state
    
    def step(self, action):
        """
            params: 
                action -  which arm to pull in our current state
            returns:
                reward -  reward 
        """
        result = np.random.randn(1)
        bandit_action = self.bandits[self.state, action]

        # if random number from the standard normal is bigger than our bandit then,
        # we get good reward
        if result > bandit_action:
            return 1
        else:
            return -1
