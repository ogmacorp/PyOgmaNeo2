// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyConstructs.h"
#include "PyComputeProgram.h"
#include "PyIntBuffer.h"
#include "PyFloatBuffer.h"
#include <ogmaneo/neo/ImageEncoder.h>
#include <fstream>

namespace pyogmaneo {
    struct PyImVisibleLayerDesc {
        PyInt3 _size;

        int _radius;

        PyImVisibleLayerDesc()
        : _size(8, 8, 16), _radius(2)
        {}

        PyImVisibleLayerDesc(const PyInt3 &size, int radius)
        : _size(size), _radius(radius)
        {}
    };

    class PyImageEncoder {
    private:
        std::vector<PyImVisibleLayerDesc> _visibleLayerDescs;

        ogmaneo::ImageEncoder _enc;

    public:
        float _alpha;

        PyImageEncoder(PyComputeSystem &cs, PyComputeProgram &prog, const PyInt3 &hiddenSize, const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs);
        PyImageEncoder(PyComputeSystem &cs, PyComputeProgram &prog, const std::string &name);

        void activate(PyComputeSystem &cs, const std::vector<PyFloatBuffer> &visibleAs);

        void learn(PyComputeSystem &cs, const std::vector<PyFloatBuffer> &visibleAs);

        void save(PyComputeSystem &cs, const std::string &name) {
            std::ofstream os(name, std::ios::binary);
            _enc.writeToStream(cs._cs, os);
        }

        int getNumVisibleLayers() const {
            return _enc.getNumVisibleLayers();
        }

        const PyImVisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        PyIntBuffer getHiddenCs() const {
            ogmaneo::Int3 size = _enc.getHiddenSize();

            PyIntBuffer buf;
            buf._size = size.x * size.y;
            buf._buf = _enc.getHiddenCs();

            return buf;
        }

        PyInt3 getHiddenSize() const {
            ogmaneo::Int3 size = _enc.getHiddenSize();

            return PyInt3(size.x, size.y, size.z);
        }
    };
}