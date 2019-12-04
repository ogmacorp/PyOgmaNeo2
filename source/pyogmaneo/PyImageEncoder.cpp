// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyogmaneo;

PyImageEncoder::PyImageEncoder(
    PyComputeSystem &cs,
    PyComputeProgram &prog,
    const PyInt3 &hiddenSize,
    const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs
) {
    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::ImageEncoder::VisibleLayerDesc> clVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        clVisibleLayerDescs[v]._size = ogmaneo::Int3(visibleLayerDescs[v]._size.x, visibleLayerDescs[v]._size.y, visibleLayerDescs[v]._size.z);
        clVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _enc.init(cs._cs, prog._prog, ogmaneo::Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z), clVisibleLayerDescs, cs._rng);
    
    _alpha = _enc._alpha;
    _minError = _enc._minError;
}

PyImageEncoder::PyImageEncoder(
    PyComputeSystem &cs,
    PyComputeProgram &prog,
    const std::string &name
) {
    std::ifstream is(name, std::ios::binary);
    _enc.readFromStream(cs._cs, prog._prog, is);

    _visibleLayerDescs.resize(_enc.getNumVisibleLayers());

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        const ogmaneo::ImageEncoder::VisibleLayerDesc &vld = _enc.getVisibleLayerDesc(v); 
        
        _visibleLayerDescs[v]._size = PyInt3(vld._size.x, vld._size.y, vld._size.z);
        _visibleLayerDescs[v]._radius = vld._radius;
    }

    _alpha = _enc._alpha;
    _minError = _enc._minError;
}

void PyImageEncoder::step(
    PyComputeSystem &cs,
    const std::vector<PyFloatBuffer> &visibleActivations,
    bool learnEnabled
) {
    _enc._alpha = _alpha;
    _enc._minError = _minError;

    std::vector<cl::Buffer> clVisibleActivations(visibleActivations.size());

    for (int v = 0; v < visibleActivations.size(); v++)
        clVisibleActivations[v] = visibleActivations[v]._buf;

    _enc.step(cs._cs, clVisibleActivations, learnEnabled);
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

    const ogmaneo::SparseMatrix &sm = _enc.getVisibleLayer(i)._weights;

    int row = ogmaneo::address3(ogmaneo::Int3(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z), _enc.getHiddenSize());
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