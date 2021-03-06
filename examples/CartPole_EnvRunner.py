# ----------------------------------------------------------------------------
#  PyOgmaNeo
#  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EOgmaNeo is licensed to you under the terms described
#  in the PYEOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

# Simple Cart-Pole example using EnvRunner

import gym
from EnvRunner import EnvRunner # EnvRunner automatically creates an OgmaNeo2 hierarchy and appropriate encoders for most Gym environments

env = gym.make('CartPole-v1')

runner = EnvRunner(env, terminalReward=-1.0, rewardScale=0.0) # Cart-Pole environment always returns a reward of 1, so use a custom reward function: -1 if episode ends, 0 otherwise

for episode in range(1000):
    env.reset()

    # Timesteps
    for t in range(500):
        done, _ = runner.act() # Step the environment and agent

        if done:
            print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))
            break