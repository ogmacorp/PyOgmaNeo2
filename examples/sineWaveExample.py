# ----------------------------------------------------------------------------
#  PyOgmaNeo
#  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EOgmaNeo is licensed to you under the terms described
#  in the PYEOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import numpy as np
import pyogmaneo
import matplotlib.pyplot as plt

# Create the compute system using a device
cs = pyogmaneo.PyComputeSystem(16)

# This defines the resolution of the input encoding - we are using a simple single column that represents a bounded scalar through a one-hot encoding. This value is the number of "bins"
inputColumnSize = 32

# The bounds of the scalar we are encoding (low, high)
bounds = (-1.0, 1.0)

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(5): # 5 layers with exponential memory
    ld = pyogmaneo.PyLayerDesc()

    # Set the hidden (encoder) layer size: width x height x columnSize
    ld._hiddenSize = pyogmaneo.PyInt3(4, 4, 16)

    ld._ffRadius = 2 # Sparse coder radius onto visible layers
    ld._fbRadius = 2 # 
    ld._pRadius = 2 # Predictor radius onto sparse coder hidden layer (and feed back)

    ld._ticksPerUpdate = 2 # How many ticks before a layer updates (compared to previous layer) - clock speed for exponential memory
    ld._temporalHorizon = 4 # Memory horizon of the layer. Must be greater or equal to ticksPerUpdate, usually equal (minimum required)

    lds.append(ld)

# Create the hierarchy: Provided with input layer sizes (a single column in this case), and input types (a single predicted layer)
h = pyogmaneo.PyHierarchy(cs, [ pyogmaneo.PyInt3(1, 1, inputColumnSize) ], [ pyogmaneo._inputTypePred ], lds)

# After creation, we can set run-time parameters
for i in range(len(lds)):
    # Encoder alpha
    h.setSCAlpha(i, 0.2)

# Present the wave sequence for some timesteps
iters = 5000

for t in range(iters):
    # The value to encode into the input column
    valueToEncode = np.sin(t * 0.02 * 2.0 * np.pi)# * np.sin(t * 0.035 * 2.0 * np.pi + 0.45)

    # Step the hierarchy given the inputs (just one here)
    h.step(cs, [ [ int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (inputColumnSize - 1) + 0.5) ] ], [], True) # True for enabling learning

    # Print progress
    if t % 100 == 0:
        print(t)

# Recall the sequence
ts = [] # Time step
vs = [] # Predicted value
trgs = [] # True value

for t in range(300):
    t2 = t + iters # Continue where previous sequence left off

    # New, continued value for comparison to what the hierarchy predicts
    valueToEncode = np.sin(t2 * 0.02 * 2.0 * np.pi)# * np.sin(t2 * 0.035 * 2.0 * np.pi + 0.45)

    # Run off of own predictions with learning disabled
    h.step(cs, [ h.getPredictionCs(0) ], [], False)

    # Retrieve the predicted column index
    predIndex = h.getPredictionCs(0)[0] # First (only in this case) input layer prediction
    
    # Decode value (de-bin)
    value = predIndex / float(inputColumnSize - 1) * (bounds[1] - bounds[0]) + bounds[0]

    # Append to plot data
    ts.append(t)
    vs.append(value)
    trgs.append(valueToEncode)

    # Show predicted value
    print(value)

# Show plot
plt.plot(ts, vs, ts, trgs)
plt.show()


