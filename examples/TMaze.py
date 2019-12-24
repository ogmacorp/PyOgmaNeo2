# ----------------------------------------------------------------------------
#  PyOgmaNeo
#  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EOgmaNeo is licensed to you under the terms described
#  in the PYEOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import pyogmaneo
from pyogmaneo import Int3
import numpy as np

# T-Maze example
# Simple prints the result of a simple T-Maze memory test environment. The more it prints just "Correct", the better it is doing.

cs = pyogmaneo.ComputeSystem("cpu", 1234, 0) # First CPU device
prog = pyogmaneo.ComputeProgram(cs, "../../OgmaNeo2/resources/neoKernels.cl")

hlds = []

# First layer description
fld = pyogmaneo.FirstLayerDesc()

fld._hiddenSize = pyogmaneo.Int3(4, 4, 16)
fld._ffRadius = 4 # Feed forward radius
fld._aRadius = 4 # Radius of action layers
fld._temporalHorizon = 2 # 2 timestep memory window. _ticksPerUpdate is always 1 for the first layer, so it is not set here
fld._historyCapacity = 32 # Reward backup horizon

# Other (higher) layer descriptions
for i in range(3): # 3 layers gives us 2^3=8 timesteps of memory
    ld = pyogmaneo.HigherLayerDesc()
    ld._hiddenSize = pyogmaneo.Int3(4, 4, 16)
    ld._ffRadius = 4 # Feed forward radius
    ld._pRadius = 4 # Prediction radius
    ld._temporalHorizon = 2 # 2 timestep memory window
    ld._ticksPerUpdate = 2 # 2 timestep striding (doubling time window every layer)

    hlds.append(ld)

h = pyogmaneo.Hierarchy(cs, prog, [ Int3(1, 1, 4), Int3(1, 1, 3) ], [ pyogmaneo._inputTypeNone, pyogmaneo._inputTypeAction ], fld, hlds)

# Buffers for holding inputs and actions
inBuf = pyogmaneo.IntBuffer(cs, 1)
aBuf = pyogmaneo.IntBuffer(cs, 1)

# Configure action layer
h.setBAAlpha(1, 0.05) # Value learning rate
h.setBABeta(1, 0.1) # Action learning rate
h.setBAGamma(1, 0.95) # Discount factor

reward = 0.0

act = 2

for i_episode in range(10000):
    mazeLen = 6 # 6-step maze

    side = int(np.random.rand() < 0.5) # Randomly choose the side the reward is on

    # Show signal
    inBuf.write(cs, [ side ])
    aBuf.write(cs, [ act ])

    h.step(cs, [ inBuf, aBuf ], True, reward)

    act = h.getActionCs(1).read(cs)[0] # Retrieve action

    # Go through tunnel
    for i in range(mazeLen):
        if act != 2: # If not taking forward action when in the beginning of the maze (tunnel)
            reward = -0.5

            print("Mistake") # Mistake while going down the tunnel
        else:
            reward = 0.0

        see = 2

        # Indicate that end of tunnel has been reached
        if i == mazeLen - 1:
            see = 3

        inBuf.write(cs, [ see ])
        aBuf.write(cs, [ act ])

        h.step(cs, [ inBuf, aBuf ], True, reward)

        act = h.getActionCs(1).read(cs)[0] # Retrieve action

    if act == side:
        reward = 1.0
        print("Correct")
    else:
        reward = -1.0
        print("Incorrect")

    act = side

    

    
