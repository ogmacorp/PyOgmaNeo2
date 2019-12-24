Examples
======================================

The `examples/ <https://github.com/ogmacorp/PyOgmaNeo2/tree/master/examples>`_ directory contains a few usage examples.

EnvRunner
**********************

The EnvRunner is a simple way to automatically create OgmaNeo2 systems for Gym tasks. It will automatically create the hierarchy and appropriate pre-encoders. This is good enough for initial experimentation, but further control requires manual usage.

CartPole example
**********************

The CartPole example is a simple test on the Gym CartPole environment. It uses EnvRunner.

TMaze example
**********************

The TMaze example is the classic T-Maze test from reinforcement learning literation. It does not use EnvRunner, so this example shows how to use the hierarchy manually.

WaveLinePrediction example
**********************

The WaveLinePrediction example shows how to use a SPH for prediction of a simple waveform. Requires matplotlib.