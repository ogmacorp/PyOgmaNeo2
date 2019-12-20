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

std::vector<float> PyHierarchy::getSCReceptiveField(
    PyComputeSystem &cs,
    int l,
    int i,
    const PyInt3 &hiddenPosition,
    PyInt3 &size
) const {
    // Determine bounds
    ogmaneo::Int3 minPos(999999, 999999, 999999);
    ogmaneo::Int3 maxPos(0, 0, 0);

    const ogmaneo::SparseMatrix &sm = _h.getSCLayer(l).getVisibleLayer(i)._weights;

    int row = ogmaneo::address3(ogmaneo::Int3(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z), _h.getSCLayer(l).getHiddenSize());
    //int nextIndex = row + 1;

    std::vector<cl_int> js(2);
    cs._cs.getQueue().enqueueReadBuffer(sm._rowRanges, CL_TRUE, row * sizeof(cl_int), 2 * sizeof(cl_int), js.data());

    int numValues = js[1] - js[0];

    if (numValues == 0)
        return {};
  
    std::vector<cl_int> columnIndices(numValues);
    cs._cs.getQueue().enqueueReadBuffer(sm._columnIndices, CL_TRUE, js[0] * sizeof(cl_int), numValues * sizeof(cl_int), columnIndices.data());

	std::vector<cl_float> nonZeroValues(numValues);
    cs._cs.getQueue().enqueueReadBuffer(sm._nonZeroValues, CL_TRUE, js[0] * sizeof(cl_int), numValues * sizeof(cl_float), nonZeroValues.data());

	for (int j = js[0]; j < js[1]; j++) {
        int index = columnIndices[j - js[0]];

        int inZ = index % _h.getSCLayer(l).getVisibleLayerDesc(i)._size.z;
        index /= _h.getSCLayer(l).getVisibleLayerDesc(i)._size.z;

        int inY = index % _h.getSCLayer(l).getVisibleLayerDesc(i)._size.y;
        index /= _h.getSCLayer(l).getVisibleLayerDesc(i)._size.y;

        int inX = index % _h.getSCLayer(l).getVisibleLayerDesc(i)._size.x;

		minPos.x = std::min(minPos.x, inX);
		minPos.y = std::min(minPos.y, inY);
		minPos.z = std::min(minPos.z, inZ);

        maxPos.x = std::max(maxPos.x, inX + 1);
		maxPos.y = std::max(maxPos.y, inY + 1);
		maxPos.z = std::max(maxPos.z, inZ + 1);
    }

    size.x = maxPos.x - minPos.x;
    size.y = maxPos.y - minPos.y;
    size.z = maxPos.z - minPos.z;

    int totalSize = size.x * size.y * size.z;
    
    std::vector<float> field(totalSize, 0.0f);

    for (int j = js[0]; j < js[1]; j++) {
        int index = columnIndices[j - js[0]];

        int inZ = index % _h.getSCLayer(l).getVisibleLayerDesc(i)._size.z;
        index /= _h.getSCLayer(l).getVisibleLayerDesc(i)._size.z;

        int inY = index % _h.getSCLayer(l).getVisibleLayerDesc(i)._size.y;
        index /= _h.getSCLayer(l).getVisibleLayerDesc(i)._size.y;

        int inX = index % _h.getSCLayer(l).getVisibleLayerDesc(i)._size.x;

		field[ogmaneo::address3(ogmaneo::Int3(inX - minPos.x, inY - minPos.y, inZ - minPos.z), ogmaneo::Int3(size.x, size.y, size.z))] = nonZeroValues[j - js[0]];
    }

    return field;
}