// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyogmaneo;

PyImageEncoder::PyImageEncoder(PyComputeSystem &cs, PyComputeProgram &prog, const PyInt3 &hiddenSize, const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs) {
    _alpha = _enc._alpha;

    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::ImageEncoder::VisibleLayerDesc> clVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        clVisibleLayerDescs[v]._size = ogmaneo::Int3(visibleLayerDescs[v]._size.x, visibleLayerDescs[v]._size.y, visibleLayerDescs[v]._size.z);
        clVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _enc.createRandom(cs._cs, prog._prog, ogmaneo::Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z), clVisibleLayerDescs, cs._rng);
}

PyImageEncoder::PyImageEncoder(PyComputeSystem &cs, PyComputeProgram &prog, const std::string &name) {
    std::ifstream is(name, std::ios::binary);
    _enc.readFromStream(cs._cs, prog._prog, is);

    _alpha = _enc._alpha;

    _visibleLayerDescs.resize(_enc.getNumVisibleLayers());

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        const ogmaneo::ImageEncoder::VisibleLayerDesc &vld = _enc.getVisibleLayerDesc(v); 
        
        _visibleLayerDescs[v]._size = PyInt3(vld._size.x, vld._size.y, vld._size.z);
        _visibleLayerDescs[v]._radius = vld._radius;
    }
}

void PyImageEncoder::activate(PyComputeSystem &cs, const std::vector<PyFloatBuffer> &visibleAs) {
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