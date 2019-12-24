Int3
======================================

.. class:: pyogmaneo.Int3

Simple 3-component integer vector type use typically to define sizes of various elements.

.. attribute:: (int32) Int3.x

    First (x) component

.. attribute:: (int32) Int3.y

    Second (y) component

.. attribute:: (int32) Int3.z

    Third (z) component

.. function:: Int3.__init__(self)

    Default initialize to x=0, y=0, z=0

.. function:: Int3.__init__(self, X, Y, Z)

    Initialize from components

    :param X: (int32) x to initialize to
    :param Y: (int32) y to initialize to
    :param Z: (int32) z to initialize to