####################################################################################################
# DO NOT MODIFY THIS FILE
####################################################################################################

import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium import ObservationWrapper

class CustomFrozenLake(FrozenLakeEnv):

    def __init__(self, slippery, render, **kwargs):
        
        custom_map = [
            "HHHHHHFFFFFFFG",
            "HHHHHFFFFFFFFF",
            "HHHHFFFFFFFFFF",
            "GFFFFHFFFFFFFF",
            "HHHHFFFFFSFFFF",
            "HHHHHFFFFFFFFF",
            "HHHHHHFFFFFFFG"]

        # create the OpenAI Gym environment
        super().__init__(desc=custom_map, is_slippery=slippery,**kwargs)

        self.slippery = slippery
        self.render_mode = "human" if render else False
        self.max_timesteps = 100
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 + 5,), dtype=np.float32
        )

        weights = np.array([-1, -4, -1, -1, -4])/100
        self.reward_func = lambda features: np.round(np.dot(features, weights),3)  # Example reward function


    def step(self, a):

        # Step return observation
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, reward, done = transitions[i]
        self.s = s
        self.ij = (self.s // self.ncol, self.s % self.ncol)
        self.lastaction = a
        if self.render_mode == "human":
            self.render()

        # Example: add [is_goal_nearby, is_on_edge] as features
        i = self.ij[0]
        j = self.ij[1]
        features = (i, j, i * j, i + j, abs(i - j))
        self.features = features

        # shape reward for better convergence
        if done and reward == 0:  # agent fell in a hole
            reward = -3.0
        elif done and reward == 1:  # agent reached the goal
            print('Reached the goal!')
            if self.ij == (3, 0):
                print("Reached the goal at (3, 0)!")
                reward = 6.0
            else:
                reward = 5.0
        else:
            reward = self.reward_func(features)

        return np.array(self.ij + features, dtype=np.float32), reward, done, False,

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        self.ij = (self.s // self.ncol, self.s % self.ncol)
        i = self.ij[0]
        j = self.ij[1]
        features = (i, j, i * j, i + j, abs(i - j))

        return np.array(self.ij + features, dtype=np.float32)


