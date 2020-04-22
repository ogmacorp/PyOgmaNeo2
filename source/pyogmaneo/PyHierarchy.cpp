// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyHierarchy.h"

using namespace pyogmaneo;

PyHierarchy::PyHierarchy(
    PyComputeSystem &cs,
    const std::vector<PyInt3> &inputSizes,
    const std::vector<int> &inputTypes,
    const std::vector<PyLayerDesc> &layerDescs
) {
    std::vector<ogmaneo::Int3> cInputSizes(inputSizes.size());

    for (int i = 0; i < inputSizes.size(); i++)
        cInputSizes[i] = ogmaneo::Int3(inputSizes[i].x, inputSizes[i].y, inputSizes[i].z);
    
    std::vector<ogmaneo::InputType> cInputTypes(inputTypes.size());

    for (int i = 0; i < inputTypes.size(); i++) {
        switch(inputTypes[i]) {
        case inputTypeNone:
            cInputTypes[i] = ogmaneo::none;
            break;
        case inputTypePrediction:
            cInputTypes[i] = ogmaneo::prediction;
            break;
        case inputTypeAction:
            cInputTypes[i] = ogmaneo::action;
            break;
        }
    }

    std::vector<ogmaneo::Hierarchy::LayerDesc> cLayerDescs(layerDescs.size());

    for (int l = 0; l < layerDescs.size(); l++) {
        cLayerDescs[l].hiddenSize = ogmaneo::Int3(layerDescs[l].hiddenSize.x, layerDescs[l].hiddenSize.y, layerDescs[l].hiddenSize.z);
        cLayerDescs[l].ffRadius = layerDescs[l].ffRadius;
        cLayerDescs[l].pRadius = layerDescs[l].pRadius;
        cLayerDescs[l].aRadius = layerDescs[l].aRadius;
        cLayerDescs[l].temporalHorizon = layerDescs[l].temporalHorizon;
        cLayerDescs[l].ticksPerUpdate = layerDescs[l].ticksPerUpdate;
        cLayerDescs[l].historyCapacity = layerDescs[l].historyCapacity;
    }

    h.initRandom(cs.cs, cInputSizes, cInputTypes, cLayerDescs);
}

PyHierarchy::PyHierarchy(
    const std::string &fileName
) {
    std::ifstream is(fileName, std::ios::binary);
    
    h.readFromStream(is);
}

void PyHierarchy::step(
    PyComputeSystem &cs,
    const std::vector<std::vector<int> > &inputCs,
    bool learnEnabled,
    float reward,
    bool mimic
) {
    assert(inputCs.size() == h.getInputSizes().size());

    std::vector<const std::vector<int>*> cInputCs(inputCs.size());

    for (int i = 0; i < inputCs.size(); i++) {
        assert(inputCs[i].size() == h.getInputSizes()[i].x * h.getInputSizes()[i].y);

        cInputCs[i] = &inputCs[i];
    }
    
    h.step(cs.cs, cInputCs, learnEnabled, reward, mimic);
}

void PyHierarchy::getState(
    PyState &state
) const {
    h.getState(state.state);
}

void PyHierarchy::setState(
    const PyState &state
) {
    h.setState(state.state);
}

void PyHierarchy::save(
    const std::string &fileName
) const {
    std::ofstream os(fileName, std::ios::binary);

    h.writeToStream(os);
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

    const ogmaneo::SparseMatrix &sm = h.getSCLayer(l).getVisibleLayer(i).weights;

    int row = ogmaneo::address3(ogmaneo::Int3(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z), h.getSCLayer(l).getHiddenSize());
    //int nextIndex = row + 1;

    std::vector<int> js(2);
    js[0] = sm.rowRanges[row];
    js[1] = sm.rowRanges[row + 1];

    int numValues = js[1] - js[0];

    if (numValues == 0)
        return {};
  
    std::vector<int> columnIndices(numValues);
    std::vector<float> nonZeroValues(numValues);

    for (int i = 0; i < numValues; i++) {
        columnIndices[i] = sm.columnIndices[js[0] + i];
        nonZeroValues[i] = sm.nonZeroValues[js[0] + i];
    }

	for (int j = js[0]; j < js[1]; j++) {
        int index = columnIndices[j - js[0]];

        int inZ = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.z;
        index /= h.getSCLayer(l).getVisibleLayerDesc(i).size.z;

        int inY = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.y;
        index /= h.getSCLayer(l).getVisibleLayerDesc(i).size.y;

        int inX = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.x;

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

        int inZ = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.z;
        index /= h.getSCLayer(l).getVisibleLayerDesc(i).size.z;

        int inY = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.y;
        index /= h.getSCLayer(l).getVisibleLayerDesc(i).size.y;

        int inX = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.x;

		field[ogmaneo::address3(ogmaneo::Int3(inX - minPos.x, inY - minPos.y, inZ - minPos.z), ogmaneo::Int3(size.x, size.y, size.z))] = nonZeroValues[j - js[0]];
    }

    return field;
}