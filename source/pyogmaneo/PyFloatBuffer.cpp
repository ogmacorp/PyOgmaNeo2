// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyFloatBuffer.h"

using namespace pyogmaneo;

void PyFloatBuffer::create(PyComputeSystem &cs, int size) {
    _size = size;

    _buf = cl::Buffer(cs._cs.getContext(), CL_MEM_READ_WRITE, _size * sizeof(cl_float));
}

void PyFloatBuffer::write(PyComputeSystem &cs, const std::vector<float> &data) {
    cs._cs.getQueue().enqueueWriteBuffer(_buf, CL_TRUE, 0, _size * sizeof(cl_float), data.data());
}

std::vector<float> PyFloatBuffer::read(PyComputeSystem &cs) const {
    std::vector<float> data(_size);

    cs._cs.getQueue().enqueueReadBuffer(_buf, CL_TRUE, 0, _size * sizeof(cl_float), data.data());

    return data;
}