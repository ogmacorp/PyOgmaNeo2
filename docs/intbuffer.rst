Int Buffer
======================================

.. class:: pyogmaneo.IntBuffer

A buffer of integers (int32). Used typically to hold CSDRs for transfering to and from the GPU/other device.

.. function:: IntBuffer.__init__(self)

   Default initialize, does nothing

.. function:: IntBuffer.__init__(self, cs, size)

   Create a buffer of a certain size (number of integers). Calls IntBuffer.create internally.

   :param cs: (ComputeSystem) compute system to use
   :param size: (int32) size of buffer (number of integers)

.. function:: IntBuffer.create(self, cs, size)

   Create a buffer of a certain size (number of integers). Will override existing data if already has some.

   :param cs: (ComputeSystem) compute system to use
   :param size: (int32) size of buffer (number of integers)

.. function:: IntBuffer.write(self, cs, data)

   Write data to the integer buffer.

   :param cs: (ComputeSystem) compute system to use
   :param data: ([int32]) list of integers to write. Must be the same size as the buffer was created with

.. function:: IntBuffer.read(self, cs)

   Read data from the integer buffer.

   :param cs: (ComputeSystem) compute system to use
   :rtype: ([int32]) list of integers

.. function:: IntBuffer.getSize(self)

   Get the size of the buffer (number of integers)

   :rtype: (int32) size