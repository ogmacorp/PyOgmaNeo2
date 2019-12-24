Float Buffer
======================================

.. class:: pyogmaneo.FloatBuffer

A buffer of floats (float32). Used typically to hold images or other input data for transfering to and from the GPU/other device.

.. function:: FloatBuffer.__init__(self)

    Default initialize, does nothing

.. function:: FloatBuffer.__init__(self, cs, size)

    Create a buffer of a certain size (number of floats). Calls FloatBuffer.create internally.

    :param cs: (ComputeSystem) compute system to use
    :param size: (int32) size of buffer (number of floats)

.. function:: FloatBuffer.create(self, cs, size)

    Create a buffer of a certain size (number of floats). Will override existing data if already has some.

    :param cs: (ComputeSystem) compute system to use
    :param size: (int32) size of buffer (number of floats)

.. function:: FloatBuffer.write(self, cs, data)

    Write data to the float buffer.

    :param cs: (ComputeSystem) compute system to use
    :param data: ([float32]) list of floats to write. Must be the same size as the buffer was created with

.. function:: FloatBuffer.read(self, cs)

    Read data from the float buffer.

    :param cs: (ComputeSystem) compute system to use
    :rtype: ([float32]) list of floats

.. function:: FloatBuffer.getSize(self)

    Get the size of the buffer (number of floats)

    :rtype: (int32) size