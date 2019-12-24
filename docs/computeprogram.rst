Compute Program
======================================

.. class:: pyogmaneo.ComputeProgram

Stores a compute program (collection of OpenCL kernels).

.. function:: ComputeProgram.__init__(self, cs, name)

    Initialize a compute program on a given compute system by reading frame a file.

    :param cs: (ComputeSystem) compute system to use
    :param name: (string) file name to load from