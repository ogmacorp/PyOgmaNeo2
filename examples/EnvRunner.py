 
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
import gym
import cv2
import os
from copy import copy

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class EnvRunner:
    def __init__(self, env, layerSizes=2 * [ Int3(4, 4, 16) ], layerRadius=4, hiddenSize=Int3(8, 8, 16), imageRadius=9, imageScale=1.0, obsResolution=32, actionResolution=16, rewardScale=1.0, terminalReward=0.0, infSensitivity=1.0, kernelPath="../OgmaNeo2/resources/neoKernels.cl", loadName=None):
        self.env = env

        self.cs = pyogmaneo.ComputeSystem("cpu", 1234, 0)
        self.prog = pyogmaneo.ComputeProgram(self.cs, kernelPath)

        self.imEnc = None
        self.imEncIndex = -1

        self.inputSizes = []
        self.inputLows = []
        self.inputHighs = []
        self.inputTypes = []
        self.imageSizes = []
        self.imageBufs = []
        self.imgsPrev = []
        self.actionIndices = []

        self.rewardScale = rewardScale
        self.terminalReward = terminalReward

        self.infSensitivity = infSensitivity

        if type(self.env.observation_space) is gym.spaces.Discrete:
            self.inputSizes.append(Int3(1, 1, self.env.observation_space.n))
            self.inputTypes.append(pyogmaneo._inputTypeNone)
            self.inputLows.append([ 0.0 ])
            self.inputHighs.append([ 0.0 ])
        elif type(self.env.observation_space) is gym.spaces.Box:
            if len(self.env.observation_space.shape) == 1 or len(self.env.observation_space.shape) == 0:
                squareSize = int(np.ceil(np.sqrt(len(self.env.observation_space.low))))
                squareTotal = squareSize * squareSize
                self.inputSizes.append(Int3(squareSize, squareSize, obsResolution))
                self.inputTypes.append(pyogmaneo._inputTypeNone)
                lows = list(self.env.observation_space.low)
                highs = list(self.env.observation_space.high)
                
                # Detect large numbers/inf
                for i in range(len(lows)):
                    if abs(lows[i]) > 100000 or abs(highs[i]) > 100000:
                        # Indicate inf by making low greater than high
                        lows[i] = 1.0
                        highs[i] = -1.0

                self.inputLows.append(lows)
                self.inputHighs.append(highs)
            elif len(self.env.observation_space.shape) == 2:
                scaledSize = ( int(self.env.observation_space.shape[0] * imageScale), int(self.env.observation_space.shape[1] * imageScale), 1 )

                self.imageSizes.append(scaledSize)
            elif len(self.env.observation_space.shape) == 3:
                scaledSize = ( int(self.env.observation_space.shape[0] * imageScale), int(self.env.observation_space.shape[1] * imageScale), 3 )

                self.imageSizes.append(scaledSize)
            else:
                raise Exception("Unsupported Box input: Dimensions too high " + str(self.env.observation_space.shape))
        else:
            raise Exception("Unsupported input type " + str(type(self.env.observation_space)))

        if len(self.imageSizes) > 0:
            vlds = []

            for i in range(len(self.imageSizes)):
                vld = pyogmaneo.ImVisibleLayerDesc(Int3(self.imageSizes[i][0], self.imageSizes[i][1], self.imageSizes[i][2]), imageRadius)

                vlds.append(vld)

                imBuf = pyogmaneo.FloatBuffer(self.cs, self.imageSizes[i][0] * self.imageSizes[i][1] * self.imageSizes[i][2])

                self.imageBufs.append(imBuf)

                self.imgsPrev.append(np.zeros(self.imageSizes[i]))

            self.imEnc = pyogmaneo.ImageEncoder(self.cs, self.prog, hiddenSize, vlds)

            self.imEncIndex = len(self.inputSizes)
            self.inputSizes.append(hiddenSize)
            self.inputTypes.append(pyogmaneo._inputTypeNone)
            self.inputLows.append([ 0.0 ])
            self.inputHighs.append([ 1.0 ])

        # Actions
        if type(self.env.action_space) is gym.spaces.Discrete:
            self.actionIndices.append(len(self.inputSizes))
            self.inputSizes.append(Int3(1, 1, self.env.action_space.n))
            self.inputTypes.append(pyogmaneo._inputTypeAction)
            self.inputLows.append([ 0.0 ])
            self.inputHighs.append([ 0.0 ])
        elif type(self.env.action_space) is gym.spaces.Box:
            if len(self.env.action_space.shape) < 3:
                if len(self.env.action_space.shape) == 2:
                    self.actionIndices.append(len(self.inputSizes))
                    self.inputSizes.append(Int3(self.env.action_space.shape[0], self.env.action_space.shape[1], actionResolution))
                    self.inputTypes.append(pyogmaneo._inputTypeAction)
                    lows = list(self.env.action_space.low)
                    highs = list(self.env.action_space.high)

                    self.inputLows.append(lows)
                    self.inputHighs.append(highs)
                else:
                    squareSize = int(np.ceil(np.sqrt(len(self.env.action_space.low))))
                    squareTotal = squareSize * squareSize
                    self.actionIndices.append(len(self.inputSizes))
                    self.inputSizes.append(Int3(squareSize, squareSize, actionResolution))
                    self.inputTypes.append(pyogmaneo._inputTypeAction)
                    lows = list(self.env.action_space.low)
                    highs = list(self.env.action_space.high)

                    self.inputLows.append(lows)
                    self.inputHighs.append(highs)
            else:
                raise Exception("Unsupported Box action: Dimensions too high " + str(self.env.action_space.shape))
        else:
            raise Exception("Unsupported action type " + str(type(self.env.action_space)))

        fld = pyogmaneo.FirstLayerDesc()

        fld._hiddenSize = layerSizes[0]
        fld._scRadius = layerRadius
        fld._lRadius = layerRadius
        fld._aRadius = layerRadius
        fld._historyCapacity = 32

        hlds = []

        for i in range(len(layerSizes) - 1):
            ld = pyogmaneo.HigherLayerDesc()
            ld._hiddenSize = layerSizes[i + 1]
            ld._ffRadius = layerRadius
            ld._lRadius = layerRadius
            ld._pRadius = layerRadius

            hlds.append(ld)

        if loadName is None:
            self.h = pyogmaneo.Hierarchy(self.cs, self.prog, self.inputSizes, self.inputTypes, fld, hlds)
        else:
            self.h = pyogmaneo.Hierarchy(self.cs, self.prog, loadName)

        self.bufs = []

        for i in range(len(self.inputSizes)):
            if i == self.imEncIndex:
                self.bufs.append(None)
            else:
                buf = pyogmaneo.IntBuffer(self.cs, self.inputSizes[i].x * self.inputSizes[i].y)
                self.bufs.append(buf)

        self.actions = []

        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            size = len(self.inputLows[index])

            startAct = []

            for j in range(size):
                startAct.append(np.random.randint(0, self.inputSizes[index].z))

            self.actions.append(startAct)

    def _feedObservation(self, obs):
        self.inputs = []

        actionIndex = 0

        for i in range(len(self.inputSizes)):
            buf = self.bufs[i]

            if self.inputTypes[i] == pyogmaneo._inputTypeAction:
                buf.write(self.cs, self.actions[actionIndex])

                actionIndex += 1
            elif i == self.imEncIndex:
                assert(buf is None)

                # Format image
                img = cv2.resize(obs, ( self.imageSizes[0][0], self.imageSizes[0][1] ))
                
                img = np.swapaxes(img, 0, 1)

                img = img.astype(np.float32) / 255.0
                
                #delta = img - self.imgsPrev[0]
 
                self.imgsPrev[0] = copy(img)

                self.imageBufs[0].write(self.cs, img.ravel().tolist())

                # Encode image
                self.imEnc.step(self.cs, [ self.imageBufs[0] ], True)

                buf = self.imEnc.getHiddenCs()

            else:
                indices = []

                for j in range(len(self.inputLows[i])):
                    if self.inputLows[i][j] < self.inputHighs[i][j]:
                        # Rescale
                        indices.append(int((obs[j] - self.inputLows[i][j]) / (self.inputHighs[i][j] - self.inputLows[i][j]) * (self.inputSizes[i].z - 1) + 0.5))
                    elif self.inputLows[i][j] > self.inputHighs[i][j]: # Inf
                        v = obs[j]

                        # Rescale
                        indices.append(int(sigmoid(v * self.infSensitivity) * (self.inputSizes[i].z - 1) + 0.5))
                    else:
                        indices.append(int(obs[j]))

                if len(indices) < self.inputSizes[i].x * self.inputSizes[i].y:
                    indices += ((self.inputSizes[i].x * self.inputSizes[i].y) - len(indices)) * [ int(0) ]

                buf.write(self.cs, indices)

            self.inputs.append(buf)

    def act(self, epsilon=0.0, obsPreprocess=None):
        feedActions = []

        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            assert(self.inputTypes[index] is pyogmaneo._inputTypeAction)

            if self.inputLows[index][0] < self.inputHighs[index][0]:
                feedAction = []

                # Explore
                for j in range(len(self.inputLows[index])):
                    if np.random.rand() < epsilon:
                        self.actions[i][j] = np.random.randint(0, self.inputSizes[index].z)

                    if self.inputLows[index][j] < self.inputHighs[index][j]:
                        feedAction.append(self.actions[i][j] / float(self.inputSizes[index].z - 1) * (self.inputHighs[index][j] - self.inputLows[index][j]) + self.inputLows[index][j])
                    else:
                        feedAction.append(self.actions[i][j])

                feedActions.append(feedAction)
            else:
                if np.random.rand() < epsilon:
                    self.actions[i][0] = np.random.randint(0, self.inputSizes[index].z)

                feedActions.append(int(self.actions[i][0]))

        # Remove outer array if needed
        if len(feedActions) == 1:
            feedActions = feedActions[0]

        obs, reward, done, info = self.env.step(feedActions)

        if obsPreprocess is not None:
            obs = obsPreprocess(obs)

        self._feedObservation(obs)

        r = reward * self.rewardScale + float(done) * self.terminalReward
        
        self.h.step(self.cs, self.inputs, True, r)

        # Retrieve actions
        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            assert(self.inputTypes[index] is pyogmaneo._inputTypeAction)

            self.actions[i] = list(self.h.getActionCs(index).read(self.cs))
        
        return done, reward

    




        
            

