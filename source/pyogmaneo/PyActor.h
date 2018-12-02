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
#include <ogmaneo/neo/Actor.h>
#include <fstream>

namespace pyogmaneo {
    struct PyAVisibleLayerDesc {
        PyInt3 _size;

        int _radius;

        PyAVisibleLayerDesc()
        : _size(8, 8, 16), _radius(2)
        {}

        PyAVisibleLayerDesc(const PyInt3 &size, int radius)
        : _size(size), _radius(radius)
        {}
    };

    class PyActor {
    private:
        std::vector<PyAVisibleLayerDesc> _visibleLayerDescs;

        ogmaneo::Actor _a;

    public:
        float _alpha;
        float _gamma;
        int _explainIters;

        PyActor(PyComputeSystem &cs, PyComputeProgram &prog, const PyInt3 &hiddenSize, int historyCapacity, const std::vector<PyAVisibleLayerDesc> &visibleLayerDescs);
        PyActor(PyComputeSystem &cs, PyComputeProgram &prog, const std::string &name);

        void step(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs, const PyIntBuffer &actionCs, float reward, bool learn);

        void save(PyComputeSystem &cs, const std::string &name) {
            std::ofstream os(name, std::ios::binary);
            _a.writeToStream(cs._cs, os);
        }

        int getNumVisibleLayers() const {
            return _a.getNumVisibleLayers();
        }

        const PyAVisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        PyIntBuffer getHiddenCs() const {
            ogmaneo::Int3 size = _a.getHiddenSize();

            PyIntBuffer buf;
            buf._size = size.x * size.y;
            buf._buf = _a.getHiddenCs();

            return buf;
        }

        PyInt3 getHiddenSize() const {
            ogmaneo::Int3 size = _a.getHiddenSize();

            return PyInt3(size.x, size.y, size.z);
        }
    };
}