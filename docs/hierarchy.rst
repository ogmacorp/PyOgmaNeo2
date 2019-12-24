Hierarchy
======================================

.. class:: pyogmaneo.Hierarchy

The sparse predictive hierarchy (SPH). Can be thought of as the "agent" when used for reinforcement learning.

.. function:: Hierarchy.__init__(self, cs, prog, inputSizes, inputTypes, firstLayerDesc, higherLayerDescs)

    Initialize a hierarchy of given structure.

    :param cs: (ComputeSystem) compute system to use
    :param prog: (ComputeProgram) compute program containing the necessary kernels (e.g. from neoKernels.cl)
    :param inputSizes: ([Int3]) list of input sizes. Defines the size of each input layer
    :param inputTypes: ([int]) list of enums defining the type of each input layer. Must be the same size as inputSizes.
        Possible enums are:
        
            pyogmaneo._inputTypeNone - used solely as an input (no additional functionality)

            pyogmaneo._inputTypePrediction - used as input and can also predict next timestep of input (t + 1)

            pyogmaneo._inputTypeAction - used as input and can also perform reinforcement learning (produces actions)

    :param firstLayerDesc: (FirstLayerDesc) A filled-out descriptor (FirstLayerDesc object) describing how the first layer in the hierarchy should look
    :param higherLayerDescs: ([HigherLayerDesc]) A list of filled-out descriptors (HigherLayerDesc objects) describing how all the higher layers (all but first) in the hierarchy should look

.. function:: Hierarchy.__init__(self, cs, prog, name)

    Initialize a hierarchy given a save file.

    :param cs: (ComputeSystem) compute system to use
    :param prog: (ComputeProgram) compute program containing the necessary kernels (e.g. from neoKernels.cl)
    :param name: (string) save file name

.. function:: Hierarchy.step(self, cs, inputCs, learnEnabled=True, reward=0.0)

    Perform a simulation step of the hierarchy. This will produce new predictions or actions if those are being used.

    :param cs: (ComputeSystem) compute system to use
    :param inputCs: ([IntBuffer]) list of input integer buffers representing the CSDRs of the dimensions described in the initialization
    :param learnEnabled: (boolean) whether or not to enable learning (if False, will only perform inference). Defaults to True
    :param reward: (float32) reward signal, if action input layers (pyogmaneo._inputTypeAction) are present this will be used to update those to maximize reward. Defaults to 0.0

.. function:: Hierarchy.save(self, cs, name)

    Save the hierarchy to a file.

    :param cs: (ComputeSystem) compute system to use
    :param name: (string) save file name

.. function:: Hierarchy.getNumLayers(self)

    Return the number of layers (encoders) the hierarchy has. Will be equal to (len(higherLayerDescs) + 1)

    :rtype: (int32) number of layers

.. function:: Hierarchy.getPredictionCs(self, i)

    Get a prediction for a given input layer index

    :param i: (int32) index of the input layer. Will error if input layer does not have type pyogmaneo._inputTypePrediction
    :rtype: (IntBuffer) integer buffer containing predictions

.. function:: Hierarchy.getActionCs(self, i)

    Get an action for a given input layer index

    :param i: (int32) index of the input layer. Will error if input layer does not have type pyogmaneo._inputTypeAction
    :rtype: (IntBuffer) integer buffer containing actions

.. function:: Hierarchy.getUpdate(self, l)

    Return whether a given layer has updated (clocked) in the last timestep

    :param l: (int32) index of the layer
    :rtype: (boolean) True if updated, False otherwise

.. function:: Hierarchy.getHiddenCs(self, l)

    Get the CSDR of the encoder of a certain layer

    :param l: (int32) index of the layer
    :rtype: (IntBuffer) integer buffer containing encoder hidden layer states

.. function:: Hierarchy.getHiddenSize(self, l)

    Get the size of the encoder of a certain layer

    :param l: (int32) index of the layer
    :rtype: (Int3) size of layer. Will always be equal to the size if was initialzed to

.. function:: Hierarchy.getTicks(self, l)

    Get current tick of a layer (clock value)

    :param l: (int32) index of the layer
    :rtype: (int32) tick value. Will always be less the value produced by Hierarchy.getTicksPerUpdate(l). Will always be 0 for the first layer (l = 0)

.. function:: Hierarchy.getTicksPerUpdate(self, l)

    Number of ticks required before a layer clocks. Typically 2

    :param l: (int32) index of the layer
    :rtype: (int32) number of ticks. Will always be 1 for the first layer (l = 0)

.. function:: Hierarchy.getNumInputLayers(self)

    Get the number of input layers. Will never deviate from the length of inputSizes given during initialization.

    :rtype: (int32) number of input layers

.. function:: Hierarchy.getNumPLayers(self, l)

    Get the number of predictor (top down) sub-layers at a given layer

    :param l: (int32) index of the layer. Must be greater than 0 as the first layer does not have regular predictors
    :rtype: (int32) number of input layers

.. function:: Hierarchy.getNumSCVisibleLayers(self, l)

    Get the number of visible (sub) layers of a sparse coder (SC), AKA encoder. This will be equal to Hierarchy.getTicksPerUpdate(l) for all l except for 0, where it will be equal to Hierarchy.getNumInputLayers()

    :param l: (int32) index of the layer
    :rtype: (int32) number of visible layers

.. function:: Hierarchy.pLayerExists(self, i)

    Determine whether there is a predictor at the i-th input layer. This will be True if the inputTypes[i] == pyogmaneo._inputTypePrediction during initialization

    :param i: (int32) index of the input layer
    :rtype: (boolean) True if exists, False otherwise

.. function:: Hierarchy.aLayerExists(self, i)

    Determine whether there is a action layer at the i-th input layer. This will be True if the inputTypes[i] == pyogmaneo._inputTypeAction during initialization

    :param i: (int32) index of the input layer
    :rtype: (boolean) True if exists, False otherwise

.. function:: Hierarchy.setSCAlpha(self, l, alpha)

    Set the alpha (learning rate) of a sparse coder (SC) (AKA encoder)

    :param l: (int32) index of the layer
    :param alpha: (float32) value to set

.. function:: Hierarchy.getSCAlpha(self, l)

    Get the alpha (learning rate) of a sparse coder (SC) (AKA encoder)

    :param l: (int32) index of the layer
    :rtype: (float32) alpha

.. function:: Hierarchy.setPAlpha(self, l, alpha)

    Set the alpha (learning rate) of a predictor (P) (AKA decoder)

    :param l: (int32) index of the layer. This function is used for predictors above the first layer, so l > 0
    :param alpha: (float32) value to set

.. function:: Hierarchy.getPAlpha(self, l)

    Get the alpha (learning rate) of a predictor (P) (AKA decoder)

    :param l: (int32) index of the layer. This function is used for predictors above the first layer, so l > 0
    :rtype: (float32) alpha

.. function:: Hierarchy.setBPAlpha(self, i, alpha)

    Set the alpha (learning rate) of a predictor (P) (AKA decoder) at the bottom (B) of the hierarchy (input layer)

    :param i: (int32) index of the input layer
    :param alpha: (float32) value to set

.. function:: Hierarchy.getBPAlpha(self, i)

    Get the alpha (learning rate) of a predictor (P) (AKA decoder) at the bottom (B) of the hierarchy (input layer)

    :param i: (int32) index of the input layer
    :rtype: (float32) alpha

.. function:: Hierarchy.setBAAlpha(self, i, alpha)

    Set the alpha (value learning rate) of an action layer (A) at the bottom (B) of the hierarchy (input layer)

    :param i: (int32) index of the input layer
    :param alpha: (float32) value to set

.. function:: Hierarchy.getBAAlpha(self, i)

    Get the alpha (value learning rate) of an action layer (A) at the bottom (B) of the hierarchy (input layer)

    :param i: (int32) index of the input layer
    :rtype: (float32) alpha

.. function:: Hierarchy.setBABeta(self, i, beta)

    Set the beta (action learning rate) of an action layer (A) at the bottom (B) of the hierarchy (input layer)

    :param i: (int32) index of the input layer
    :param beta: (float32) value to set

.. function:: Hierarchy.getBABeta(self, i)

    Get the beta (action learning rate) of an action layer (A) at the bottom (B) of the hierarchy (input layer)

    :param i: (int32) index of the input layer
    :rtype: (float32) beta

.. function:: Hierarchy.setBAGamma(self, i, gamma)

    Set the gamma (discount factor) of an action layer (A) at the bottom (B) of the hierarchy (input layer)

    :param i: (int32) index of the input layer
    :param beta: (float32) value to set

.. function:: Hierarchy.getBAGamma(self, i)

    Get the gamma (discount factor) of an action layer (A) at the bottom (B) of the hierarchy (input layer)

    :param i: (int32) index of the input layer
    :rtype: (float32) gamma
    