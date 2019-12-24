Compute System
======================================

.. class:: pyogmaneo.ComputeSystem

Contains context information for accelerating compution.

.. function:: ComputeSystem.__init__(self, type="gpu", seed=1234, platformIndex=-1, deviceIndex=-1)
    
    Initialize the compute system.

    :param type: (string) device type, can be "cpu", "gpu", or "all"
    :param seed: (uint64) random number generator seed
    :param platformIndex: (int32) index of platform. Set to -1 to use default
    :param deviceIndex: (int32) index of device. Set to -1 to use default