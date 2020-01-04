// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyogmaneo;

PyImageEncoder::PyImageEncoder(
    PyComputeSystem &cs,
    const PyInt3 &hiddenSize,
    const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs
) {
    std::vector<ogmaneo::ImageEncoder::VisibleLayerDesc> cVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        cVisibleLayerDescs[v]._size = ogmaneo::Int3(visibleLayerDescs[v]._size.x, visibleLayerDescs[v]._size.y, visibleLayerDescs[v]._size.z);
        cVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _enc.initRandom(cs._cs, ogmaneo::Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z), cVisibleLayerDescs);

    _alpha = _enc._alpha;
    _epsilon = _enc._epsilon;
}

PyImageEncoder::PyImageEncoder(
    const std::string &fileName
) {
    std::ifstream is(fileName, std::ios::binary);
    
    _enc.readFromStream(is);

    _alpha = _enc._alpha;
    _epsilon = _enc._epsilon;
}

void PyImageEncoder::activate(
    PyComputeSystem &cs,
    const std::vector<std::vector<float> > &visibleActivations
) {
    _enc._alpha = _alpha;
    _enc._epsilon = _epsilon;
    
    std::vector<const std::vector<float>*> cVisibleActivations(visibleActivations.size());

    for (int i = 0; i < visibleActivations.size(); i++)
        cVisibleActivations[i] = &visibleActivations[i];

    _enc.activate(cs._cs, cVisibleActivations);
}

void PyImageEncoder::learn(
    PyComputeSystem &cs,
    const std::vector<std::vector<float> > &visibleActivations
) {
    _enc._alpha = _alpha;
    _enc._epsilon = _epsilon;
    
    std::vector<const std::vector<float>*> cVisibleActivations(visibleActivations.size());

    for (int i = 0; i < visibleActivations.size(); i++)
        cVisibleActivations[i] = &visibleActivations[i];

    _enc.learn(cs._cs, cVisibleActivations);
}

void PyImageEncoder::reconstruct(
    PyComputeSystem &cs,
    const std::vector<int> &hiddenCs
) {
    _enc.reconstruct(cs._cs, &hiddenCs);
}

void PyImageEncoder::save(
    const std::string &fileName
) const {
    std::ofstream os(fileName, std::ios::binary);

    _enc.writeToStream(os);
}

std::vector<float> PyImageEncoder::getReceptiveField(
    PyComputeSystem &cs,
    int i,
    const PyInt3 &hiddenPosition,
    PyInt3 &size
) const {
    // Determine bounds
    ogmaneo::Int3 minPos(999999, 999999, 999999);
    ogmaneo::Int3 maxPos(0, 0, 0);

    const ogmaneo::SparseMatrix &sm = _enc.getVisibleLayer(i)._decWeights;

    int row = ogmaneo::address3(ogmaneo::Int3(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z), _enc.getHiddenSize());
    //int nextIndex = row + 1;

    std::vector<int> js(2);
    js[0] = sm._rowRanges[row];
    js[1] = sm._rowRanges[row + 1];

    int numValues = js[1] - js[0];

    if (numValues == 0)
        return {};
  
    std::vector<int> columnIndices(numValues);
    std::vector<float> nonZeroValues(numValues);

    for (int i = 0; i < numValues; i++) {
        columnIndices[i] = sm._columnIndices[js[0] + i];
        nonZeroValues[i] = sm._nonZeroValues[js[0] + i];
    }
    
	for (int j = js[0]; j < js[1]; j++) {
        int index = columnIndices[j - js[0]];

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
    
    std::vector<float> field(totalSize, 0.0f);

    for (int j = js[0]; j < js[1]; j++) {
        int index = columnIndices[j - js[0]];

        int inZ = index % _enc.getVisibleLayerDesc(i)._size.z;
        index /= _enc.getVisibleLayerDesc(i)._size.z;

        int inY = index % _enc.getVisibleLayerDesc(i)._size.y;
        index /= _enc.getVisibleLayerDesc(i)._size.y;

        int inX = index % _enc.getVisibleLayerDesc(i)._size.x;

		field[ogmaneo::address3(ogmaneo::Int3(inX - minPos.x, inY - minPos.y, inZ - minPos.z), ogmaneo::Int3(size.x, size.y, size.z))] = nonZeroValues[j - js[0]];
    }

    return field;
}