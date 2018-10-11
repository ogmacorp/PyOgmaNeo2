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

cs = pyogmaneo.PyComputeSystem()

# NOTE: Copy neoKernels.cl from your OgmaNeo2 repository to this directory!
prog = pyogmaneo.PyComputeProgram(cs, "neoKernels.cl")

inputColumnSize = 64

bounds = (-1.0, 1.0) # Range of value

lds = []

for i in range(9):
    ld = pyogmaneo.PyLayerDesc()

    ld._hiddenSize = pyogmaneo.PyInt3(4, 4, 16)
    
    lds.append(ld)

h = pyogmaneo.PyHierarchy(cs, prog, [ pyogmaneo.PyInt3(1, 1, inputColumnSize) ], [ pyogmaneo._inputTypePredict ], lds)

for i in range(len(lds)):
    h.setPAlpha(i, 0, 0.5) # Set predictor alpha to 0.5 for all layers (and the only predictor for the inputs)

ioBuf = pyogmaneo.PyIntBuffer(cs, 1)

fbSize = lds[-1]._hiddenSize.x * lds[-1]._hiddenSize.y

topFB = pyogmaneo.PyIntBuffer(cs, fbSize)
topFB.write(cs, fbSize * [ 0 ])

# Present the wave sequence
iters = 3000

for t in range(iters):
    index = t

    valueToEncode = np.sin(index * 0.02 * 2.0 * np.pi)

    ioBuf.write(cs, [ int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (inputColumnSize - 1) + 0.5) ])

    h.step(cs, [ ioBuf ], topFB, True)

    if t % 100 == 0:
        print(t)

# Recall
ts = []
vs = []
trgs = []

for t in range(300):
    index = t + iters

    valueToEncode = np.sin(index * 0.02 * 2.0 * np.pi)

    ioBuf.write(cs, [ int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (inputColumnSize - 1) + 0.5) ])

    h.step(cs, [ h.getPredictionCs(0) ], topFB, False)

    predIndex = h.getPredictionCs(0).read(cs)[0] # First (only in this case) input layer prediction
    
    # Decode value
    value = predIndex / float(inputColumnSize - 1) * (bounds[1] - bounds[0]) + bounds[0]

    ts.append(t)
    vs.append(value)
    trgs.append(valueToEncode)

    print(value)

plt.plot(ts, vs, ts, trgs)
plt.show()


