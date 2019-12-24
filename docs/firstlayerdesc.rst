First Layer Descriptor
======================================

.. class:: pyogmaneo.FirstLayerDesc

Describes the first (bottom-most) layer of the hierarchy during initialization

.. attribute:: (Int3) FirstLayerDesc._hiddenSize

    Hidden layer sparse coder (AKA encoder) size

.. attribute:: (int32) FirstLayerDesc._ffRadius

    Feed-forward (bottom-up) sparse coder (AKA encoder) radius. Must be 0 or greater

.. attribute:: (int32) FirstLayerDesc._pRadius

    Feed-back (top-down) predictor radius. Must be 0 or greater

.. attribute:: (int32) FirstLayerDesc._aRadius

    Action radius. Must be 0 or greater

.. attribute:: (int32) FirstLayerDesc._temporalHorizon

    Temporal horizon of sparse coder (AKA encoder). Must be greater than or equal to 1

.. attribute:: (int32) FirstLayerDesc._historyCapacity

    History capacity of action layers. Used to propagate reward back through time. 32 is a good value to start with

.. function:: FirstLayerDesc.__init__(self)

    Initialize to sensible defaults

.. function:: FirstLayerDesc.__init__(self, hiddenSize, ffRadius, pRadius, aRadius, temporalHorizon, historyCapacity)

    Initialize to given values