// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PySparseCoder.h"

using namespace pyogmaneo;

PySparseCoder::PySparseCoder(PyComputeSystem &cs, PyComputeProgram &prog, std::tuple<int, int, int> hiddenSize, const std::vector<PySCVisibleLayerDesc> &visibleLayerDescs) {
    _alpha = _sc._alpha;
    _explainIters = _sc._explainIters;

    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::SparseCoder::VisibleLayerDesc> clVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        clVisibleLayerDescs[v]._size = cl_int3{ std::get<0>(visibleLayerDescs[l]._size), std::get<1>(visibleLayerDescs[l]._size), std::get<2>(visibleLayerDescs[l]._size) };
        clVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _sc.createRandom(cs._cs, prog._prog, cl_int3{ std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize) }, clVisibleLayerDescs, cs._rng);
}

void PySparseCoder::activate(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs) {
    _sc._explainIters = _explainIters;

    std::vector<cl::Buffer> clVisibleCs(visibleCs.size());

    for (int v = 0; v < visibleCs.size(); v++)
        clVisibleCs[v] = visibleCs[v]._buf;

    _sc.activate(cs._cs, clVisibleCs);
}

void PySparseCoder::learn(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs) {
    _sc._alpha = _alpha;
    
    std::vector<cl::Buffer> clVisibleCs(visibleCs.size());

    for (int v = 0; v < visibleCs.size(); v++)
        clVisibleCs[v] = visibleCs[v]._buf;

    _sc.learn(cs._cs, clVisibleCs);
}