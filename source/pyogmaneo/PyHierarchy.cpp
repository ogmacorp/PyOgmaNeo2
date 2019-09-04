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
        cLayerDescs[l]._rfRadius = layerDescs[l]._rfRadius;
        cLayerDescs[l]._rrRadius = layerDescs[l]._rrRadius;
        cLayerDescs[l]._pRadius = layerDescs[l]._pRadius;
        cLayerDescs[l]._rfScale = layerDescs[l]._rfScale;
        cLayerDescs[l]._rfDropRatio = layerDescs[l]._rfDropRatio;
        cLayerDescs[l]._rrScale = layerDescs[l]._rrScale;
        cLayerDescs[l]._rrDropRatio = layerDescs[l]._rrDropRatio;
        cLayerDescs[l]._pDropRatio = layerDescs[l]._pDropRatio;
    }

    _h.initRandom(cs._cs, cInputSizes, cInputTypes, cLayerDescs);
}

PyHierarchy::PyHierarchy(const std::string &fileName) {
    std::ifstream is(fileName, std::ios::binary);
    
    _h.readFromStream(is);
}

void PyHierarchy::step(PyComputeSystem &cs, const std::vector<std::vector<float> > &inputStates, const std::vector<float> &goalStates, bool learn) {
    assert(inputStates.size() == _h.getInputSizes().size());

    std::vector<const std::vector<float>*> cInputStates(inputStates.size());

    for (int i = 0; i < inputStates.size(); i++) {
        assert(inputStates[i].size() == _h.getInputSizes()[i].x * _h.getInputSizes()[i].y);

        cInputStates[i] = &inputStates[i];
    }
    
    _h.step(cs._cs, cInputStates, &goalStates, learn);
}

void PyHierarchy::save(const std::string &fileName) const {
    std::ofstream os(fileName, std::ios::binary);

    _h.writeToStream(os);
}