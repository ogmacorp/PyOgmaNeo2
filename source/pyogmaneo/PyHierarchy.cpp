// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyHierarchy.h"

using namespace pyogmaneo;

PyHierarchy::PyHierarchy(
    PyComputeSystem &cs,
    PyComputeProgram &prog,
    const std::vector<PyInt3> &inputSizes,
    const std::vector<int> &inputTypes,
    const PyFirstLayerDesc &firstLayerDesc,
    const std::vector<PyHigherLayerDesc> &higherLayerDescs
) {
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
        case _inputTypeAct:
            clInputTypes[i] = ogmaneo::_act;
            break;
        }
    }

    ogmaneo::Hierarchy::FirstLayerDesc clFirstLayerDesc;

    clFirstLayerDesc._hiddenSize = ogmaneo::Int3(firstLayerDesc._hiddenSize.x, firstLayerDesc._hiddenSize.y, firstLayerDesc._hiddenSize.z);
    clFirstLayerDesc._ffRadius = firstLayerDesc._ffRadius;
    clFirstLayerDesc._aRadius = firstLayerDesc._aRadius;
    clFirstLayerDesc._temporalHorizon = firstLayerDesc._temporalHorizon;
    clFirstLayerDesc._historyCapacity = firstLayerDesc._historyCapacity;

    std::vector<ogmaneo::Hierarchy::HigherLayerDesc> clHigherLayerDescs(higherLayerDescs.size());

    for (int l = 0; l < higherLayerDescs.size(); l++) {
        clHigherLayerDescs[l]._hiddenSize = ogmaneo::Int3(higherLayerDescs[l]._hiddenSize.x, higherLayerDescs[l]._hiddenSize.y, higherLayerDescs[l]._hiddenSize.z);
        clHigherLayerDescs[l]._ffRadius = higherLayerDescs[l]._ffRadius;
        clHigherLayerDescs[l]._pRadius = higherLayerDescs[l]._pRadius;
        clHigherLayerDescs[l]._temporalHorizon = higherLayerDescs[l]._temporalHorizon;
        clHigherLayerDescs[l]._ticksPerUpdate = higherLayerDescs[l]._ticksPerUpdate;
    }

    _h.init(cs._cs, prog._prog, clInputSizes, clInputTypes, clFirstLayerDesc, clHigherLayerDescs, cs._rng);
}

PyHierarchy::PyHierarchy(
    PyComputeSystem &cs,
    PyComputeProgram &prog,
    const std::string &name
) {
    std::ifstream is(name, std::ios::binary);
    _h.readFromStream(cs._cs, prog._prog, is);

    _inputSizes.resize(_h.getInputSizes().size());

    for (int i = 0; i < _inputSizes.size(); i++)
        _inputSizes[i] = PyInt3(_h.getInputSizes()[i].x, _h.getInputSizes()[i].y, _h.getInputSizes()[i].z);
}

void PyHierarchy::step(
    PyComputeSystem &cs,
    const std::vector<PyIntBuffer> &inputCs,
    float reward,
    bool learn
) {
    std::vector<cl::Buffer> clInputCs(inputCs.size());

    for (int i = 0; i < inputCs.size(); i++)
        clInputCs[i] = inputCs[i]._buf;

    _h.step(cs._cs, clInputCs, cs._rng, reward, learn);
}
