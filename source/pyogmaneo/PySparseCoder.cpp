// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PySparseCoder.h"

using namespace pyogmaneo;

PySparseCoder::PySparseCoder(PyComputeSystem &cs, PyComputeProgram &prog, const PyInt3 &hiddenSize, const std::vector<PySCVisibleLayerDesc> &visibleLayerDescs) {
    _alpha = _sc._alpha;
    _explainIters = _sc._explainIters;

    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::SparseCoder::VisibleLayerDesc> clVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        clVisibleLayerDescs[v]._size = cl_int3{ visibleLayerDescs[v]._size.x, visibleLayerDescs[v]._size.y, visibleLayerDescs[v]._size.z };
        clVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _sc.createRandom(cs._cs, prog._prog, cl_int3{ hiddenSize.x, hiddenSize.y, hiddenSize.z }, clVisibleLayerDescs, cs._rng);
}

PySparseCoder::PySparseCoder(PyComputeSystem &cs, PyComputeProgram &prog, const std::string &name) {
    std::ifstream is(name, std::ios::binary);
    _sc.readFromStream(cs._cs, prog._prog, is);

    _alpha = _sc._alpha;
    _explainIters = _sc._explainIters;

    _visibleLayerDescs.resize(_sc.getNumVisibleLayers());

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        const ogmaneo::SparseCoder::VisibleLayerDesc &vld = _sc.getVisibleLayerDesc(v); 
        
        _visibleLayerDescs[v]._size = PyInt3(vld._size.x, vld._size.y, vld._size.z);
        _visibleLayerDescs[v]._radius = vld._radius;
    }
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