Image Encoder
======================================

.. class:: pyogmaneo.ImageEncoder

The ImageEncoder is a pre-encoder used to convert images to CSDRs.

.. attribute:: (float32) ImageEncoder._alpha

    Learning rate

.. attribute:: (float32) ImageEncoder._gamma

    SOM falloff

.. function:: ImageEncoder.__init__(self, cs, prog, hiddenSize, visibleLayerDescs)

    Initialize an image encoder of given structure.

    :param cs: (ComputeSystem) compute system to use
    :param prog: (ComputeProgram) compute program containing the necessary kernels (e.g. from neoKernels.cl)
    :param hiddenSize: (Int3) size of the output (hidden) layer
    :param visibleLayerDescs: ([ImVisibleLayerDesc]) list of ImVisibleLayerDesc describing each input (visible) layer

.. function:: ImageEncoder.__init__(self, cs, prog, name)

    Initialize an image encoder given a save file.

    :param cs: (ComputeSystem) compute system to use
    :param prog: (ComputeProgram) compute program containing the necessary kernels (e.g. from neoKernels.cl)
    :param name: (string) save file name

.. function:: ImageEncoder.step(self, cs, visibleActivations, learnEnabled=True)

    Perform a simulation step of the hierarchy. This will produce new predictions or actions if those are being used.

    :param cs: (ComputeSystem) compute system to use
    :param visibleActivations: ([FloatBuffer]) list of input float buffers representing the image of the dimensions described in the initialization
    :param learnEnabled: (boolean) whether or not to enable learning (if False, will only perform inference). Defaults to True

.. function:: ImageEncoder.save(self, cs, name)

    Save the image encoder to a file.

    :param cs: (ComputeSystem) compute system to use
    :param name: (string) save file name

.. function:: ImageEncoder.getNumVisibleLayers(self)

    Return the number of visible (input) layers the image encoder has. Will be equal to len(visibleLayerDescs) provided during initialization

    :rtype: (int32) number of layers

.. function:: ImageEncoder.getVisibleLayerDesc(self, index)

    Retrieve the ImVisibleLayerDesc used to initialize the image encoder at a certain index

    :param index: (int32) index of the input (visible) layer
    :rtype: (ImVisibleLayerDesc) the descriptor

.. function:: ImageEncoder.getHiddenCs(self)

    Get the hidden encoded state (output CSDR)

    :rtype: (IntBuffer) the CSDR

.. function:: ImageEncoder.getHiddenSize(self)

    Get the size of the hidden state

    :rtype: (Int3) the CSDR size
