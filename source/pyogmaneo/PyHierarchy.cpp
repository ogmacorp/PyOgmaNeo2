// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyHierarchy.h"

using namespace pyogmaneo;

PyHierarchy::PyHierarchy(PyComputeSystem &cs, const std::vector<PyInt2> &inputSizes, const std::vector<PyLayerDesc> &layerDescs) {
    std::vector<ogmaneo::Int2> cInputSizes(inputSizes.size());

    for (int i = 0; i < inputSizes.size(); i++)
        cInputSizes[i] = ogmaneo::Int2(inputSizes[i].x, inputSizes[i].y);


    std::vector<ogmaneo::Hierarchy::LayerDesc> cLayerDescs(layerDescs.size());

    for (int l = 0; l < layerDescs.size(); l++) {
        cLayerDescs[l]._hiddenSize = ogmaneo::Int2(layerDescs[l]._hiddenSize.x, layerDescs[l]._hiddenSize.y);
        cLayerDescs[l]._scRadius = layerDescs[l]._scRadius;
        cLayerDescs[l]._pRadius = layerDescs[l]._pRadius;
        cLayerDescs[l]._temporalHorizon = layerDescs[l]._temporalHorizon;
        cLayerDescs[l]._ticksPerUpdate = layerDescs[l]._ticksPerUpdate;
    }

    _h.initRandom(cs._cs, cInputSizes, cLayerDescs);
}

PyHierarchy::PyHierarchy(const std::string &fileName) {
    std::ifstream is(fileName, std::ios::binary);
    
    _h.readFromStream(is);
}

void PyHierarchy::step(PyComputeSystem &cs, const std::vector<std::vector<float> > &inputs, bool learn) {
    assert(inputs.size() == _h.getInputSizes().size());

    std::vector<const std::vector<float>*> cInputs(inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        assert(inputs[i].size() == _h.getInputSizes()[i].x * _h.getInputSizes()[i].y);

        cInputs[i] = &inputs[i];
    }
    
    _h.step(cs._cs, cInputs, learn);
}

void PyHierarchy::save(const std::string &fileName) const {
    std::ofstream os(fileName, std::ios::binary);

    _h.writeToStream(os);
}