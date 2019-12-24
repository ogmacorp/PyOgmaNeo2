Higher Layer Descriptor
======================================

.. class:: pyogmaneo.HigherLayerDesc

Describes the higher (all but bottom-most) layers of the hierarchy during initialization

.. attribute:: (Int3) HigherLayerDesc._hiddenSize

    Hidden layer sparse coder (AKA encoder) size

.. attribute:: (int32) HigherLayerDesc._ffRadius

    Feed-forward (bottom-up) sparse coder (AKA encoder) radius. Must be 0 or greater

.. attribute:: (int32) HigherLayerDesc._pRadius

    Feed-back (top-down) predictor radius. Must be 0 or greater

.. attribute:: (int32) HigherLayerDesc._ticksPerUpdate

    Temporal horizon of sparse coder (AKA encoder). Must be greater than or equal to 1

.. attribute:: (int32) HigherLayerDesc._temporalHorizon

    Temporal horizon of sparse coder (AKA encoder). Must be greater than or equal to HigherLayerDesc._ticksPerUpdate

.. function:: HigherLayerDesc.__init__(self)

   	Initialize to sensible defaults

.. function:: HigherLayerDesc.__init__(self, hiddenSize, ffRadius, pRadius, ticksPerUpdate, temporalHorizon)

   	Initialize to given values