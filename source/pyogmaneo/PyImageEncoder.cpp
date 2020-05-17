// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
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
        cVisibleLayerDescs[v].size = ogmaneo::Int3(visibleLayerDescs[v].size.x, visibleLayerDescs[v].size.y, visibleLayerDescs[v].size.z);
        cVisibleLayerDescs[v].radius = visibleLayerDescs[v].radius;
    }

    enc.initRandom(cs.cs, ogmaneo::Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z), cVisibleLayerDescs);

    alpha = enc.alpha;
    beta = enc.beta;
    gamma = enc.gamma;
}

PyImageEncoder::PyImageEncoder(
    const std::string &fileName
) {
    std::ifstream is(fileName, std::ios::binary);
    
    enc.readFromStream(is);

    alpha = enc.alpha;
    beta = enc.beta;
    gamma = enc.gamma;
}

void PyImageEncoder::step(
    PyComputeSystem &cs,
    const std::vector<std::vector<float> > &visibleActivations,
    bool learnEnabled
) {
    enc.alpha = alpha;
    enc.beta = beta;
    enc.gamma = gamma;
    
    std::vector<const std::vector<float>*> cVisibleActivations(visibleActivations.size());

    for (int i = 0; i < visibleActivations.size(); i++)
        cVisibleActivations[i] = &visibleActivations[i];

    enc.step(cs.cs, cVisibleActivations, learnEnabled);
}

void PyImageEncoder::reconstruct(
    PyComputeSystem &cs,
    const std::vector<int> &hiddenCs
) {
    enc.reconstruct(cs.cs, &hiddenCs);
}

void PyImageEncoder::save(
    const std::string &fileName
) const {
    std::ofstream os(fileName, std::ios::binary);

    enc.writeToStream(os);
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

    const ogmaneo::SparseMatrix &sm = enc.getVisibleLayer(i).weights;

    int row = ogmaneo::address3(ogmaneo::Int3(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z), enc.getHiddenSize());
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

        int inZ = index % enc.getVisibleLayerDesc(i).size.z;
        index /= enc.getVisibleLayerDesc(i).size.z;

        int inY = index % enc.getVisibleLayerDesc(i).size.y;
        index /= enc.getVisibleLayerDesc(i).size.y;

        int inX = index % enc.getVisibleLayerDesc(i).size.x;

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

        int inZ = index % enc.getVisibleLayerDesc(i).size.z;
        index /= enc.getVisibleLayerDesc(i).size.z;

        int inY = index % enc.getVisibleLayerDesc(i).size.y;
        index /= enc.getVisibleLayerDesc(i).size.y;

        int inX = index % enc.getVisibleLayerDesc(i).size.x;

		field[ogmaneo::address3(ogmaneo::Int3(inX - minPos.x, inY - minPos.y, inZ - minPos.z), ogmaneo::Int3(size.x, size.y, size.z))] = nonZeroValues[j - js[0]];
    }

    return field;
}