 
import pyogmaneo
from pyogmaneo import PyInt3
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import lycon
from copy import copy

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

inputTypeNone = pyogmaneo._inputTypeNone
inputTypeAct = pyogmaneo._inputTypeAct

class EnvRunner:
    def __init__(self, env, layerSizes=3 * [ PyInt3(3, 3, 32) ], layerRadius=3, hiddenSize=PyInt3(8, 8, 32), imageRadius=6, imageScale=0.5, resolution=32, rewardScale=1.0, terminalReward=0.0, infSensitivity=2.0, rewardClip=0.0):
        self.env = env

        self.cs = pyogmaneo.PyComputeSystem(16, 1234)

        self.imEnc = None
        self.imEncIndex = -1

        self.inputSizes = []
        self.inputLows = []
        self.inputHighs = []
        self.inputTypes = []
        self.imageSizes = []
        self.imagesPrev = []
        self.actionIndices = []

        self.rewardScale = rewardScale
        self.terminalReward = terminalReward
        self.rewardClip = rewardClip

        self.infSensitivity = infSensitivity

        if type(self.env.observation_space) is gym.spaces.Discrete:
            self.inputSizes.append(PyInt3(1, 1, self.env.observation_space.n))
            self.inputTypes.append(inputTypeNone)
            self.inputLows.append([ 0.0 ])
            self.inputHighs.append([ 0.0 ])
        elif type(self.env.observation_space) is gym.spaces.Box:
            if len(self.env.observation_space.shape) == 1 or len(self.env.observation_space.shape) == 0:
                squareSize = int(np.ceil(np.sqrt(len(self.env.observation_space.low))))
                squareTotal = squareSize * squareSize
                self.inputSizes.append(PyInt3(squareSize, squareSize, resolution))
                self.inputTypes.append(inputTypeNone)
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
                self.imagesPrev.append(np.zeros(scaledSize))
            elif len(self.env.observation_space.shape) == 3:
                scaledSize = ( int(self.env.observation_space.shape[0] * imageScale), int(self.env.observation_space.shape[1] * imageScale), 3 )

                self.imageSizes.append(scaledSize)
                self.imagesPrev.append(np.zeros(scaledSize))
            else:
                raise Exception("Unsupported Box input: Dimensions too high " + str(self.env.observation_space.shape))
        else:
            raise Exception("Unsupported input type " + str(type(self.env.observation_space)))

        if len(self.imageSizes) > 0:
            vlds = []

            for i in range(len(self.imageSizes)):
                vld = pyogmaneo.PyImVisibleLayerDesc(PyInt3(self.imageSizes[i][0], self.imageSizes[i][1], self.imageSizes[i][2]), imageRadius)

                vlds.append(vld)

            self.imEnc = pyogmaneo.PyImageEncoder(self.cs, hiddenSize, vlds)

            self.imEncIndex = len(self.inputSizes)
            self.inputSizes.append(hiddenSize)
            self.inputTypes.append(inputTypeNone)
            self.inputLows.append([ 0.0 ])
            self.inputHighs.append([ 1.0 ])

        # Actions
        if type(self.env.action_space) is gym.spaces.Discrete:
            self.actionIndices.append(len(self.inputSizes))
            self.inputSizes.append(PyInt3(1, 1, self.env.action_space.n))
            self.inputTypes.append(inputTypeAct)
            self.inputLows.append([ 0.0 ])
            self.inputHighs.append([ 0.0 ])
        elif type(self.env.action_space) is gym.spaces.Box:
            if len(self.env.action_space.shape) < 3:
                if len(self.env.action_space.shape) == 2:
                    self.actionIndices.append(len(self.inputSizes))
                    self.inputSizes.append(PyInt3(self.env.action_space.shape[0], self.env.action_space.shape[1], resolution))
                    self.inputTypes.append(inputTypeAct)
                    lows = list(self.env.action_space.low)
                    highs = list(self.env.action_space.high)

                    self.inputLows.append(lows)
                    self.inputHighs.append(highs)
                else:
                    squareSize = int(np.ceil(np.sqrt(len(self.env.action_space.low))))
                    squareTotal = squareSize * squareSize
                    self.actionIndices.append(len(self.inputSizes))
                    self.inputSizes.append(PyInt3(squareSize, squareSize, resolution))
                    self.inputTypes.append(inputTypeAct)
                    lows = list(self.env.action_space.low)
                    highs = list(self.env.action_space.high)

                    self.inputLows.append(lows)
                    self.inputHighs.append(highs)
            else:
                raise Exception("Unsupported Box action: Dimensions too high " + str(self.env.action_space.shape))
        else:
            raise Exception("Unsupported action type " + str(type(self.env.action_space)))

        lds = []

        for i in range(len(layerSizes)):
            ld = pyogmaneo.PyLayerDesc()
            ld._hiddenSize = layerSizes[i]
            #ld._oneHotSize = oneHotSize
            ld._scRadius = layerRadius
            ld._aRadius = layerRadius
            ld._rRadius = layerRadius
            ld._historyCapacity = 16

            lds.append(ld)

        self.h = pyogmaneo.PyHierarchy(self.cs, self.inputSizes, self.inputTypes, lds)

        # for i in range(len(layerSizes)):
        #     #self.h.setSCAlpha(i, 0.1)

        #     for j in range(self.h.getNumVisibleLayers(i)):
        #         if self.h.visibleLayerExists(i, j):
        #             self.h.setAAlpha(i, j, 0.005)

        self.actions = []

        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            size = self.inputSizes[index].x * self.inputSizes[index].y

            startAct = []

            for j in range(size):
                startAct.append(np.random.randint(0, self.inputSizes[index].z))

            self.actions.append(startAct)

        # Get starting observation
        #self._feedObservation(self.env.reset())

    def _feedObservation(self, obs):
        self.inputs = []

        actionIndex = 0

        for i in range(len(self.inputSizes)):
            if self.inputTypes[i] == inputTypeAct:
                self.inputs.append(self.actions[actionIndex])

                actionIndex += 1
            elif i == self.imEncIndex:
                # Format image
                img = lycon.resize(obs, width=self.imageSizes[0][1], height=self.imageSizes[0][0], interpolation=lycon.Interpolation.CUBIC)

                #img = np.swapaxes(img, 0, 1) / 255.0
                img = img.astype(np.float32) / 255.0

                # delta = img - self.imagesPrev

                self.imagesPrev[0] = copy(img)

                img = img.ravel().astype(np.float32).tolist()
                
                # Encode image
                self.imEnc.step(self.cs, [ list(img) ], True)

                self.inputs.append(list(self.imEnc.getHiddenCs()))

                # vals = list(self.imEnc.getHiddenCs())

                # os.system("clear")

                # for y in range(self.imEnc.getHiddenSize().y):
                #     s = ""

                #     for x in range(self.imEnc.getHiddenSize().x):
                #         v = vals[x + y * self.imEnc.getHiddenSize().x]
                #         s += str(v) + ("  " if v < 10 else " ")

                #     print(s)

            else:
                indices = []

                for j in range(len(self.inputLows[i])):
                    if self.inputLows[i][j] < self.inputHighs[i][j]:
                        # Rescale
                        indices.append(int((obs[j] - self.inputLows[i][j]) / (self.inputHighs[i][j] - self.inputLows[i][j]) * (self.inputSizes[i].z - 1) + 0.5))
                    elif self.inputLows[i][j] > self.inputHighs[i][j]: # Inf
                        # Rescale
                        indices.append(int(sigmoid(obs[j] * self.infSensitivity) * (self.inputSizes[i].z - 1) + 0.5))
                        #print(indices[-1])
                    else:
                        indices.append(int(obs[j]))

                if len(indices) < self.inputSizes[i].x * self.inputSizes[i].y:
                    indices += ((self.inputSizes[i].x * self.inputSizes[i].y) - len(indices)) * [ int(0) ]

                self.inputs.append(indices)

    def act(self, epsilon=0.0, obsPreprocess=None):
        feedActions = []

        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            assert(self.inputTypes[index] is inputTypeAct)

            if self.inputLows[index][0] < self.inputHighs[index][0]:
                feedAction = []

                # Explore
                for j in range(len(self.inputLows[index])):
                    if np.random.rand() < epsilon:
                        self.actions[i][j] = np.random.randint(0, self.inputSizes[index].z)

                    if self.inputLows[index][j] < self.inputHighs[index][j]:
                        feedAction.append(self.actions[i][j] / float(self.inputSizes[index].z) * (self.inputHighs[index][j] - self.inputLows[index][j]) + self.inputLows[index][j])
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

        if self.rewardClip != 0.0:
            r = min(self.rewardClip, max(-self.rewardClip, r))

        self.h.step(self.cs, self.inputs, r, True)
        
        # vals = list(self.h.getHiddenCs(0))

        # os.system("clear")

        # for y in range(self.h.getHiddenSize(0).y):
        #     s = ""

        #     for x in range(self.h.getHiddenSize(0).x):
        #         v = vals[x + y * self.h.getHiddenSize(0).x]
        #         s += str(v) + ("  " if v < 10 else " ")

        #     print(s)

        # Retrieve actions
        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            assert(self.inputTypes[index] is inputTypeAct)

            self.actions[i] = list(self.h.getActionCs(index))
      
        return done, reward

    




        
            

