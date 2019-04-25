import pyogmaneo

import numpy as np
import gym
import gym_ple
import lycon # Image resizing library
from copy import copy
import scipy.misc
from scipy import ndimage, misc
import cv2
import losswise
import os

env = gym.make('Pong-v0')

imageSize = ( 16, 16 )
encHiddenSize = ( 8, 8, 32 )
numActions = env.action_space.n

cs = pyogmaneo.PyComputeSystem(16)

lds = []

numLayers = 2

for i in range(numLayers):
    ld = pyogmaneo.PyLayerDesc()
    ld._hiddenSize = pyogmaneo.PyInt3(5, 5, 32)
    
    ld._scRadius = 3
    ld._aRadius = 3

    lds.append(ld)

h = pyogmaneo.PyHierarchy(cs, [ pyogmaneo.PyInt3(encHiddenSize[0], encHiddenSize[1], encHiddenSize[2]), pyogmaneo.PyInt3(1, 1, numActions) ], [ pyogmaneo._inputTypeNone, pyogmaneo._inputTypeAct ], lds)

enc = pyogmaneo.PyImageEncoder(cs, pyogmaneo.PyInt3(encHiddenSize[0], encHiddenSize[1], encHiddenSize[2]), [ pyogmaneo.PyImVisibleLayerDesc(pyogmaneo.PyInt3(imageSize[0], imageSize[1], 1), 6) ])

minSize = min(env.observation_space.shape[0], env.observation_space.shape[1])
maxSize = max(env.observation_space.shape[0], env.observation_space.shape[1])
cropHeightOffset = 8

obsPrev = np.zeros((minSize, minSize, 3), dtype=np.float32)

print("Num actions: " + str(numActions))

reward = 0.0
averageReward = 0.0
rNoise = 0.0

#cv2.namedWindow('image', cv2.WINDOW_NORMAL)

episodeCount = 20000

for episode in range(episodeCount):
    try:
        obs = env.reset()

        totalReward = 0.0

        for t in range(10000):
            #if episode % 100 == 99:
            env.render()

            obs = obs.astype(dtype=np.float32) / 255.0#np.swapaxes(obs.astype(dtype=np.float32) / 255.0, 0, 1)

            obs = obs[maxSize // 2 - minSize // 2 + cropHeightOffset : maxSize // 2 + minSize // 2 + cropHeightOffset, :, :]
            
            deltaObs = obs - obsPrev

            obsPrev = copy(obs)

            deltaObs = (deltaObs[:, :, 0] + deltaObs[:, :, 1] + deltaObs[:, :, 2]) / 3.0

            #obs = (obs[:, :, 0] + obs[:, :, 1] + obs[:, :, 2]) / 3.0
            deltaObs = ndimage.sobel(deltaObs)
           
            deltaObs = lycon.resize(deltaObs, width=imageSize[0], height=imageSize[1], interpolation=lycon.Interpolation.LANCZOS)
            deltaObs = np.tanh(deltaObs * 10.0) * 1.0
            #print(obs)
            
            #showObs = ((deltaObs * 0.5 + 0.5) * 255).astype(np.uint8)
            #showObs = cv2.resize(showObs, (256, 256))

            #cv2.imshow('image', showObs)
            ##scipy.misc.toimage(obs, cmin=0.0, cmax=1.0).save('outfile.jpg')
            #cv2.waitKey(1)

            enc.step(cs, [ deltaObs.ravel().tolist() ], True)

            # os.system("clear")

            # vals = list(enc.getHiddenCs())

            # for y in range(enc.getHiddenSize().y):
            #     s = ""

            #     for x in range(enc.getHiddenSize().x):
            #         v = vals[x + y * enc.getHiddenSize().x]
            #         s += str(v) + ("  " if v < 10 else " ")

            #     print(s)
    
            h.step(cs, [ enc.getHiddenCs(), h.getActionCs(1) ], reward, True)
            
            action = int(h.getActionCs(1)[0])

            #print(h.getOutputStates())

            obs, reward, done, info = env.step(action)

            totalReward += reward

            if done:
                break

        if episode == 0:
            averageReward = totalReward
        else:
            averageReward = 0.9 * averageReward + 0.1 * totalReward
    
        print("Episode {} finished after {} timesteps, gathering {} reward. Average reward: {}".format(episode + 1, t + 1, totalReward, averageReward))
    
    except KeyboardInterrupt:
        print("Stopped by user.")
        break

print("Done.")

env.close()