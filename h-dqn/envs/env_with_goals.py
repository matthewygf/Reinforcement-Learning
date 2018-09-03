from envs.ale_atari_env import AtariEnv
import numpy as np

class EnvWithGoals(AtariEnv):
    def __init__(self, monitor):
        super(EnvWithGoals, self).__init__(monitor)

        self.goals = []
        self.goals_reached = [False, False, False, False]

    def set_goals(self, goals):
        self.goals = goals
    
    def add_goal(self, goal):
        self.goals.append(goal)



