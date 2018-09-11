import gym
import numpy as np

class Critic(object):
    def __init__(self, goal_env):
        assert isinstance(goal_env, gym.GoalEnv)
        self.env = goal_env

    # only assign intrinsic reward if the goal is reached 
    # and it has not been reached previously
    def criticize(self, 
                  desired_goal_t,
                  reached_goal_t,
                  primitive_action_t, 
                  done_t,
                  distance_reward=0.0):
        reward = 0.0
        # if the goal has been reached before, let's return 0 reward
        achieved_before = desired_goal_t in self.env.goals_history
        if achieved_before:
            return reward
        
        if reached_goal_t:
            reward += 1.0
        
        # because agent died,
        # gives him -1
        if done_t:
            reward -= 1.0

        reward+=distance_reward
        
        # clip intrinsic reward
        reward = np.minimum(reward, 1)
        reward = np.maximum(reward, -1)
        return reward
        


        