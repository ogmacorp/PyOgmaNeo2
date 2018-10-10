// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyActor.h"

using namespace pyogmaneo;

PyActor::PyActor(PyComputeSystem &cs, PyComputeProgram &prog, std::tuple<int, int, int> hiddenSize, const std::vector<PyAVisibleLayerDesc> &visibleLayerDescs) {
    _alpha = _a._alpha;
    _gamma = _a._gamma;

    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::Actor::VisibleLayerDesc> clVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        clVisibleLayerDescs[v]._size = cl_int3{ std::get<0>(visibleLayerDescs[l]._size), std::get<1>(visibleLayerDescs[l]._size), std::get<2>(visibleLayerDescs[l]._size) };
        clVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _p.createRandom(cs._cs, prog._prog, cl_int3{ std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize) }, clVisibleLayerDescs, cs._rng);
}

void PyActor::step(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs, const PyIntBuffer &targetCs, float reward, bool learn) {
    std::vector<cl::Buffer> clVisibleCs(visibleCs.size());

    for (int v = 0; v < visibleCs.size(); v++)
        clVisibleCs[v] = visibleCs[v]._buf;

    _a.step(cs._cs, clVisibleCs, targetCs._buf, reward, learn);
}