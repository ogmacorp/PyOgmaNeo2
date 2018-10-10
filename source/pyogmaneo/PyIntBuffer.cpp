// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyIntBuffer.h"

using namespace pyogmaneo;

void PyIntBuffer::create(PyComputeSystem &cs, int size) {
    _size = size;

    _buf = cl::Buffer(cs._cs.getContext(), CL_MEM_READ_WRITE, _size * sizeof(cl_int));
}

void PyIntBuffer::write(PyComputeSystem &cs, const std::vector<int> &data) {
    cs._cs.getQueue().enqueueWriteBuffer(_buf, CL_TRUE, 0, _size * sizeof(cl_int), data.data());
}

std::vector<int> PyIntBuffer::read(PyComputeSystem &cs) const {
    std::vector<int> data(_size);

    cs._cs.getQueue().enqueueReadBuffer(_buf, CL_TRUE, 0, _size * sizeof(cl_int), data.data());

    return data;
}