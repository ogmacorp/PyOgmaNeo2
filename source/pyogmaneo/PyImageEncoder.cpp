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
    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::ImageEncoder::VisibleLayerDesc> cVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        cVisibleLayerDescs[v]._size = ogmaneo::Int3(visibleLayerDescs[v]._size.x, visibleLayerDescs[v]._size.y, visibleLayerDescs[v]._size.z);
        cVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _enc.createRandom(cs._cs, ogmaneo::Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z), cVisibleLayerDescs);
}

void PyImageEncoder::step(PyComputeSystem &cs, const std::vector<std::vector<float> > &visibleActivations, bool learnEnabled) {
    std::vector<const std::vector<float>*> cVisibleActivations(visibleActivations.size());

    for (int i = 0; i < visibleActivations.size(); i++) {
        cVisibleActivations[i] = &visibleActivations[i];
    }

    _enc.step(cs._cs, cVisibleActivations, learnEnabled);
}