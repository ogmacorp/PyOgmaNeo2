// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyogmaneo;

PyImageEncoder::PyImageEncoder(PyComputeSystem &cs, PyComputeProgram &prog, std::tuple<int, int, int> hiddenSize, const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs) {
    _alpha = _enc._alpha;
    _explainIters = _enc._explainIters;

    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::ImageEncoder::VisibleLayerDesc> clVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        clVisibleLayerDescs[v]._size = cl_int3{ std::get<0>(visibleLayerDescs[l]._size), std::get<1>(visibleLayerDescs[l]._size), std::get<2>(visibleLayerDescs[l]._size) };
        clVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _enc.createRandom(cs._cs, prog._prog, cl_int3{ std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize) }, clVisibleLayerDescs, cs._rng);
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