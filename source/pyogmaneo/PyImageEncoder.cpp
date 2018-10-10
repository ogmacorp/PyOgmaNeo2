// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyogmaneo;

PyImageEncoder::PyImageEncoder(PyComputeSystem &cs, PyComputeProgram &prog, std::array<int, 3> hiddenSize, const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs) {
    _alpha = _enc._alpha;
    _explainIters = _enc._explainIters;

    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::ImageEncoder::VisibleLayerDesc> clVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        clVisibleLayerDescs[v]._size = cl_int3{ visibleLayerDescs[v]._size[0], visibleLayerDescs[v]._size[1], visibleLayerDescs[v]._size[2] };
        clVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _enc.createRandom(cs._cs, prog._prog, cl_int3{ hiddenSize[0], hiddenSize[1], hiddenSize[2] }, clVisibleLayerDescs, cs._rng);
}

void PyImageEncoder::activate(PyComputeSystem &cs, const std::vector<PyFloatBuffer> &visibleAs) {
    _enc._explainIters = _explainIters;

    std::vector<cl::Buffer> clVisibleAs(visibleAs.size());

    for (int v = 0; v < visibleAs.size(); v++)
        clVisibleAs[v] = visibleAs[v]._buf;

    _enc.activate(cs._cs, clVisibleAs);
}

void PyImageEncoder::learn(PyComputeSystem &cs, const std::vector<PyFloatBuffer> &visibleAs) {
    _enc._alpha = _alpha;
    
    std::vector<cl::Buffer> clVisibleAs(visibleAs.size());

    for (int v = 0; v < visibleAs.size(); v++)
        clVisibleAs[v] = visibleAs[v]._buf;

    _enc.learn(cs._cs, clVisibleAs);
}