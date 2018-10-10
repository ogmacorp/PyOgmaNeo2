// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyHierarchy.h"

using namespace pyogmaneo;

PyHierarchy::PyHierarchy(PyComputeSystem &cs, PyComputeProgram &prog, const std::vector<std::tuple<int, int, int> > &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs) {
    std::vector<cl_int3> clInputSizes(inputSizes.size());
    
    for (int i = 0; i < inputSizes.size(); i++)
        clInputSizes[i] = cl_int3{ std::get<0>(inputSizes[i]), std::get<1>(inputSizes[i]), std::get<2>(inputSizes[i]) };

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
        clLayerDescs[l]._hiddenSize = cl_int3{ std::get<0>(layerDescs[l]._hiddenSize), std::get<1>(layerDescs[l]._hiddenSize), std::get<2>(layerDescs[l]._hiddenSize) };
        clLayerDescs[l]._scRadius = layerDescs[l]._scRadius;
        clLayerDescs[l]._pRadius = layerDescs[l]._pRadius;
        clLayerDescs[l]._temporalHorizon = layerDescs[l]._temporalHorizon;
        clLayerDescs[l]._ticksPerUpdate = layerDescs[l]._ticksPerUpdate;
    }

    _h.createRandom(cs._cs, prog._prog, clInputSizes, clInputTypes, clLayerDescs, cs._rng);
}

void PyHierarchy::step(PyComputeSystem &cs, const std::vector<PyIntBuffer> &inputCs, const PyIntBuffer &topFeedBack, bool learn, float reward) {
    std::vector<cl::Buffer> clInputCs(inputCs.size());

    for (int i = 0; i < inputCs.size(); i++)
        clInputCs[i] = inputCs[i]._buf;

    _h.step(cs._cs, clInputCs, topFeedBack._buf, learn, reward);
}
