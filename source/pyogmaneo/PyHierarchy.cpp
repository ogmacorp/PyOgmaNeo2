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
        cLayerDescs[l]._lRadius = layerDescs[l]._lRadius;
        cLayerDescs[l]._temporalHorizon = layerDescs[l]._temporalHorizon;
        cLayerDescs[l]._ticksPerUpdate = layerDescs[l]._ticksPerUpdate;
    }

    _h.initRandom(cs._cs, cInputSizes, cInputTypes, cLayerDescs);
}

PyHierarchy::PyHierarchy(const std::string &fileName) {
    std::ifstream is(fileName, std::ios::binary);
    
    _h.readFromStream(is);
}

void PyHierarchy::step(PyComputeSystem &cs, const std::vector<std::vector<int> > &inputCs, const std::vector<int> &topFeedBackCs, bool learn) {
    assert(inputCs.size() == _h.getInputSizes().size());

    std::vector<const std::vector<int>*> cInputCs(inputCs.size());

    for (int i = 0; i < inputCs.size(); i++) {
        assert(inputCs[i].size() == _h.getInputSizes()[i].x * _h.getInputSizes()[i].y);

        cInputCs[i] = &inputCs[i];
    }
    
    _h.step(cs._cs, cInputCs, &topFeedBackCs, learn);
}

void PyHierarchy::save(const std::string &fileName) const {
    std::ofstream os(fileName, std::ios::binary);

    _h.writeToStream(os);
}

void PyHierarchy::getSCReceptiveField(int l, int i, const PyInt3 &hiddenPosition, std::vector<float> &field, PyInt3 &size) const {
    // Determine bounds
    ogmaneo::Int3 minPos(999999, 999999, 999999);
    ogmaneo::Int3 maxPos(0, 0, 0);

    const ogmaneo::SparseMatrix sm = _h.getSCLayer(l).getWeights(i);

    int row = ogmaneo::address3(ogmaneo::Int3(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z), _h.getSCLayer(l).getHiddenSize());
    int nextIndex = row + 1;
	
	for (int j = sm._rowRanges[row]; j < sm._rowRanges[nextIndex]; j++) {
        int index = sm._columnIndices[j];

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

    field.clear();
    field.resize(totalSize, 0.0f);

    for (int j = sm._rowRanges[row]; j < sm._rowRanges[nextIndex]; j++) {
        int index = sm._columnIndices[j];

        int inZ = index % _h.getSCLayer(l).getVisibleLayerDesc(i)._size.z;
        index /= _h.getSCLayer(l).getVisibleLayerDesc(i)._size.z;

        int inY = index % _h.getSCLayer(l).getVisibleLayerDesc(i)._size.y;
        index /= _h.getSCLayer(l).getVisibleLayerDesc(i)._size.y;

        int inX = index % _h.getSCLayer(l).getVisibleLayerDesc(i)._size.x;

		field[ogmaneo::address3(ogmaneo::Int3(inX - minPos.x, inY - minPos.y, inZ - minPos.z), ogmaneo::Int3(size.x, size.y, size.z))] = sm._nonZeroValues[j];
    }
}