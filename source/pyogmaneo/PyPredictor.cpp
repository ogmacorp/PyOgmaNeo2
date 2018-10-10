// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyPredictor.h"

using namespace pyogmaneo;

PyPredictor::PyPredictor(PyComputeSystem &cs, PyComputeProgram &prog, std::array<int, 3> hiddenSize, const std::vector<PyPVisibleLayerDesc> &visibleLayerDescs) {
    _alpha = _p._alpha;

    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::Predictor::VisibleLayerDesc> clVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        clVisibleLayerDescs[v]._size = cl_int3{ visibleLayerDescs[v]._size[0], visibleLayerDescs[v]._size[1], visibleLayerDescs[v]._size[2] };
        clVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _p.createRandom(cs._cs, prog._prog, cl_int3{ hiddenSize[0], hiddenSize[1], hiddenSize[2] }, clVisibleLayerDescs, cs._rng);
}

void PyPredictor::activate(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs) {
    std::vector<cl::Buffer> clVisibleCs(visibleCs.size());

    for (int v = 0; v < visibleCs.size(); v++)
        clVisibleCs[v] = visibleCs[v]._buf;

    _p.activate(cs._cs, clVisibleCs);
}

void PyPredictor::learn(PyComputeSystem &cs, const PyIntBuffer &targetCs) {
    _p._alpha = _alpha;

    _p.learn(cs._cs, targetCs._buf);
}