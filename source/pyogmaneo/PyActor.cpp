// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyActor.h"

using namespace pyogmaneo;

PyActor::PyActor(PyComputeSystem &cs, PyComputeProgram &prog, const PyInt3 &hiddenSize, int historyCapacity, const std::vector<PyAVisibleLayerDesc> &visibleLayerDescs) {
    _alpha = _a._alpha;
    _gamma = _a._gamma;
    _explainIters = _a._explainIters;

    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::Actor::VisibleLayerDesc> clVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        clVisibleLayerDescs[v]._size = ogmaneo::Int3(visibleLayerDescs[v]._size.x, visibleLayerDescs[v]._size.y, visibleLayerDescs[v]._size.z);
        clVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _a.createRandom(cs._cs, prog._prog, ogmaneo::Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z), historyCapacity, clVisibleLayerDescs, cs._rng);
}

PyActor::PyActor(PyComputeSystem &cs, PyComputeProgram &prog, const std::string &name) {
    std::ifstream is(name, std::ios::binary);
    _a.readFromStream(cs._cs, prog._prog, is);

    _alpha = _a._alpha;
    _gamma = _a._gamma;
    _explainIters = _a._explainIters;

    _visibleLayerDescs.resize(_a.getNumVisibleLayers());

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        const ogmaneo::Actor::VisibleLayerDesc &vld = _a.getVisibleLayerDesc(v); 
        
        _visibleLayerDescs[v]._size = PyInt3(vld._size.x, vld._size.y, vld._size.z);
        _visibleLayerDescs[v]._radius = vld._radius;
    }
}

void PyActor::step(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs, const PyIntBuffer &actionCs, float reward, bool learn) {
    _a._alpha = _alpha;
    _a._gamma = _gamma;
    _a._explainIters = _explainIters;

    std::vector<cl::Buffer> clVisibleCs(visibleCs.size());

    for (int v = 0; v < visibleCs.size(); v++)
        clVisibleCs[v] = visibleCs[v]._buf;

    _a.step(cs._cs, clVisibleCs, actionCs._buf, cs._rng, reward, learn);
}