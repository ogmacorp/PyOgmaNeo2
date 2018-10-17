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
        float _traceDecay;
        float _tdErrorClip;

        PyActor(PyComputeSystem &cs, PyComputeProgram &prog, const PyInt3 &hiddenSize, const std::vector<PyAVisibleLayerDesc> &visibleLayerDescs);
        PyActor(PyComputeSystem &cs, PyComputeProgram &prog, const std::string &name) {
            std::ifstream is(name, std::ios::binary);
            _a.readFromStream(cs._cs, prog._prog, is);

            _alpha = _a._alpha;
            _gamma = _a._gamma;
            _traceDecay = _a._traceDecay;
            _tdErrorClip = _a._tdErrorClip;
        }

        void step(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs, const PyIntBuffer &targetCs, float reward, bool learn);

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
            cl_int3 size = _a.getHiddenSize();

            PyIntBuffer buf;
            buf._size = size.x * size.y;
            buf._buf = _a.getHiddenCs();

            return buf;
        }

        PyInt3 getHiddenSize() const {
            cl_int3 size = _a.getHiddenSize();

            return PyInt3(size.x, size.y, size.z);
        }
    };
}