// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyogmaneo;

PyImageEncoder::PyImageEncoder(PyComputeSystem &cs, const PyInt3 &hiddenSize, int lRadius, const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs) {
    std::vector<ogmaneo::ImageEncoder::VisibleLayerDesc> cVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        cVisibleLayerDescs[v]._size = ogmaneo::Int3(visibleLayerDescs[v]._size.x, visibleLayerDescs[v]._size.y, visibleLayerDescs[v]._size.z);
        cVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _enc.initRandom(cs._cs, ogmaneo::Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z), lRadius, cVisibleLayerDescs);

    _explainIters = _enc._explainIters;
}

PyImageEncoder::PyImageEncoder(const std::string &fileName) {
    std::ifstream is(fileName, std::ios::binary);
    
    _enc.readFromStream(is);

    _explainIters = _enc._explainIters;
}

void PyImageEncoder::step(PyComputeSystem &cs, const std::vector<std::vector<float> > &visibleActivations, bool learnEnabled) {
    _enc._explainIters = _explainIters;
    
    std::vector<const std::vector<float>*> cVisibleActivations(visibleActivations.size());

    for (int i = 0; i < visibleActivations.size(); i++) {
        cVisibleActivations[i] = &visibleActivations[i];
    }

    _enc.step(cs._cs, cVisibleActivations, learnEnabled);
}

void PyImageEncoder::reconstruct(PyComputeSystem &cs, const std::vector<int> &hiddenCs) {
    _enc.reconstruct(cs._cs, &hiddenCs);
}

void PyImageEncoder::save(const std::string &fileName) const {
    std::ofstream os(fileName, std::ios::binary);

    _enc.writeToStream(os);
}

void PyImageEncoder::getReceptiveField(int i, const PyInt3 &hiddenPosition, std::vector<float> &field, PyInt3 &size) const {
    // Determine bounds
    ogmaneo::Int3 minPos(999999, 999999, 999999);
    ogmaneo::Int3 maxPos(0, 0, 0);

    const ogmaneo::SparseMatrix sm = _enc.getWeights(i);

    int row = ogmaneo::address3(ogmaneo::Int3(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z), _enc.getHiddenSize());
    int nextIndex = row + 1;
	
	for (int j = sm._rowRanges[row]; j < sm._rowRanges[nextIndex]; j++) {
        int index = sm._columnIndices[j];

        int inZ = index % _enc.getVisibleLayerDesc(i)._size.z;
        index /= _enc.getVisibleLayerDesc(i)._size.z;

        int inY = index % _enc.getVisibleLayerDesc(i)._size.y;
        index /= _enc.getVisibleLayerDesc(i)._size.y;

        int inX = index % _enc.getVisibleLayerDesc(i)._size.x;

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

        int inZ = index % _enc.getVisibleLayerDesc(i)._size.z;
        index /= _enc.getVisibleLayerDesc(i)._size.z;

        int inY = index % _enc.getVisibleLayerDesc(i)._size.y;
        index /= _enc.getVisibleLayerDesc(i)._size.y;

        int inX = index % _enc.getVisibleLayerDesc(i)._size.x;

		field[ogmaneo::address3(ogmaneo::Int3(inX - minPos.x, inY - minPos.y, inZ - minPos.z), ogmaneo::Int3(size.x, size.y, size.z))] = sm._nonZeroValues[j];
    }
}