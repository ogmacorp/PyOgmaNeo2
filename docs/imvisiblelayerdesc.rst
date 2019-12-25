Image Visible Layer Desciptor
======================================

.. class:: pyogmaneo.ImVisibleLayerDesc

Describes the first (bottom-most) layer of the hierarchy during initialization

.. attribute:: (Int3) ImVisibleLayerDesc._size

    Size of the input. Last value (z) will be 3 for RGB images

.. attribute:: (int32) ImVisibleLayerDesc._radius

    Radius onto input. Must be 0 or greater

.. function:: ImVisibleLayerDesc.__init__(self)

    Initialize to sensible defaults

.. function:: ImVisibleLayerDesc.__init__(self, size, radius)

    Initialize to given values