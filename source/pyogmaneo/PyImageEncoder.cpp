// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyogmaneo;

PyImageEncoder::PyImageEncoder(PyComputeSystem &cs, const PyInt3 &hiddenSize, const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs) {
    std::vector<ogmaneo::ImageEncoder::VisibleLayerDesc> cVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        cVisibleLayerDescs[v]._size = ogmaneo::Int3(visibleLayerDescs[v]._size.x, visibleLayerDescs[v]._size.y, visibleLayerDescs[v]._size.z);
        cVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _enc.initRandom(cs._cs, ogmaneo::Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z), cVisibleLayerDescs);

    _alpha = _enc._alpha;
    _beta = _enc._beta;
    _explainIters = _enc._explainIters;
}

PyImageEncoder::PyImageEncoder(const std::string &fileName) {
    std::ifstream is(fileName, std::ios::binary);
    
    _enc.readFromStream(is);
    
    _alpha = _enc._alpha;
    _beta = _enc._beta;
    _explainIters = _enc._explainIters;
}

void PyImageEncoder::step(PyComputeSystem &cs, const std::vector<std::vector<float> > &visibleActivations, bool learnEnabled) {
    _enc._alpha = _alpha;
    _enc._beta = _beta;
    _enc._explainIters = _explainIters;

    std::vector<const std::vector<float>*> cVisibleActivations(visibleActivations.size());

    for (int i = 0; i < visibleActivations.size(); i++) {
        cVisibleActivations[i] = &visibleActivations[i];
    }

    _enc.step(cs._cs, cVisibleActivations, learnEnabled);
}

void PyImageEncoder::reconstruct(PyComputeSystem &cs, const std::vector<int> &hiddenCs) {
    _enc.reconstruct(cs._cs, &hiddenCs);
}

void PyImageEncoder::save(const std::string &fileName) const {
    std::ofstream os(fileName, std::ios::binary);

    _enc.writeToStream(os);
}