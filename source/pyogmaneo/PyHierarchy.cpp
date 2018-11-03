// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyHierarchy.h"

using namespace pyogmaneo;

PyHierarchy::PyHierarchy(PyComputeSystem &cs, PyComputeProgram &prog, const std::vector<PyInt3> &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs) {
    _inputSizes = inputSizes;
    
    std::vector<ogmaneo::Int3> clInputSizes(inputSizes.size());
    
    for (int i = 0; i < inputSizes.size(); i++)
        clInputSizes[i] = ogmaneo::Int3(inputSizes[i].x, inputSizes[i].y, inputSizes[i].z);

    std::vector<ogmaneo::InputType> clInputTypes(inputTypes.size());

    for (int i = 0; i < inputTypes.size(); i++) {
        switch(inputTypes[i]) {
        case _inputTypeNone:
            clInputTypes[i] = ogmaneo::_none;
            break;
        case _inputTypePredict:
            clInputTypes[i] = ogmaneo::_predict;
            break;
        case _inputTypeAct:
            clInputTypes[i] = ogmaneo::_act;
            break;
        }
    }

    std::vector<ogmaneo::Hierarchy::LayerDesc> clLayerDescs(layerDescs.size());

    for (int l = 0; l < layerDescs.size(); l++) {
        clLayerDescs[l]._hiddenSize = ogmaneo::Int3(layerDescs[l]._hiddenSize.x, layerDescs[l]._hiddenSize.y, layerDescs[l]._hiddenSize.z);
        clLayerDescs[l]._scRadius = layerDescs[l]._scRadius;
        clLayerDescs[l]._aRadius = layerDescs[l]._aRadius;
        clLayerDescs[l]._temporalHorizon = layerDescs[l]._temporalHorizon;
        clLayerDescs[l]._ticksPerUpdate = layerDescs[l]._ticksPerUpdate;
        clLayerDescs[l]._historyCapacity = layerDescs[l]._historyCapacity;
    }

    _h.createRandom(cs._cs, prog._prog, clInputSizes, clInputTypes, clLayerDescs, cs._rng);
}

PyHierarchy::PyHierarchy(PyComputeSystem &cs, PyComputeProgram &prog, const std::string &name) {
    std::ifstream is(name, std::ios::binary);
    _h.readFromStream(cs._cs, prog._prog, is);

    _inputSizes.resize(_h.getInputSizes().size());

    for (int i = 0; i < _inputSizes.size(); i++)
        _inputSizes[i] = PyInt3(_h.getInputSizes()[i].x, _h.getInputSizes()[i].y, _h.getInputSizes()[i].z);
}

void PyHierarchy::step(PyComputeSystem &cs, const std::vector<PyIntBuffer> &inputCs, bool learn, float reward) {
    std::vector<cl::Buffer> clInputCs(inputCs.size());

    for (int i = 0; i < inputCs.size(); i++)
        clInputCs[i] = inputCs[i]._buf;

    _h.step(cs._cs, clInputCs, cs._rng, learn, reward);
}
