// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyHierarchy.h"

using namespace pyogmaneo;

PyHierarchy::PyHierarchy(PyComputeSystem &cs, const std::vector<PyInt3> &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs) {
    std::vector<ogmaneo::Int3> cInputSizes(inputSizes.size());

    for (int i = 0; i < inputSizes.size(); i++)
        cInputSizes[i] = ogmaneo::Int3(inputSizes[i].x, inputSizes[i].y, inputSizes[i].z);
    
    std::vector<ogmaneo::InputType> cInputTypes(inputTypes.size());

    for (int i = 0; i < inputTypes.size(); i++) {
        switch(inputTypes[i]) {
        case _inputTypeNone:
            cInputTypes[i] = ogmaneo::_none;
            break;
        case _inputTypePred:
            cInputTypes[i] = ogmaneo::_predict;
            break;
        }
    }

    std::vector<ogmaneo::Hierarchy::LayerDesc> cLayerDescs(layerDescs.size());

    for (int l = 0; l < layerDescs.size(); l++) {
        cLayerDescs[l]._hiddenSize = ogmaneo::Int3(layerDescs[l]._hiddenSize.x, layerDescs[l]._hiddenSize.y, layerDescs[l]._hiddenSize.z);
        cLayerDescs[l]._scRadius = layerDescs[l]._scRadius;
        cLayerDescs[l]._pRadius = layerDescs[l]._pRadius;
        cLayerDescs[l]._temporalHorizon = layerDescs[l]._temporalHorizon;
        cLayerDescs[l]._ticksPerUpdate = layerDescs[l]._ticksPerUpdate;
    }

    _h.initRandom(cs._cs, cInputSizes, cInputTypes, cLayerDescs);
}

PyHierarchy::PyHierarchy(const std::string &fileName) {
    std::ifstream is(fileName, std::ios::binary);
    
    _h.readFromStream(is);
}

void PyHierarchy::step(PyComputeSystem &cs, const std::vector<std::vector<int> > &inputCs, bool learnEnabled) {
    assert(inputCs.size() == _h.getInputSizes().size());

    std::vector<const std::vector<int>*> cInputCs(inputCs.size());

    for (int i = 0; i < inputCs.size(); i++) {
        assert(inputCs[i].size() == _h.getInputSizes()[i].x * _h.getInputSizes()[i].y);

        cInputCs[i] = &inputCs[i];
    }
    
    _h.step(cs._cs, cInputCs, learnEnabled);
}

void PyHierarchy::save(const std::string &fileName) const {
    std::ofstream os(fileName, std::ios::binary);

    _h.writeToStream(os);
}