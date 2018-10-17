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
#include <ogmaneo/neo/Predictor.h>
#include <fstream>

namespace pyogmaneo {
    struct PyPVisibleLayerDesc {
        PyInt3 _size;

        int _radius;

        PyPVisibleLayerDesc()
        : _size(8, 8, 16), _radius(2)
        {}

        PyPVisibleLayerDesc(const PyInt3 &size, int radius)
        : _size(size), _radius(radius)
        {}
    };

    class PyPredictor {
    private:
        std::vector<PyPVisibleLayerDesc> _visibleLayerDescs;

        ogmaneo::Predictor _p;

    public:
        float _alpha;

        PyPredictor(PyComputeSystem &cs, PyComputeProgram &prog, const PyInt3 &hiddenSize, const std::vector<PyPVisibleLayerDesc> &visibleLayerDescs);
        PyPredictor(PyComputeSystem &cs, PyComputeProgram &prog, const std::string &name) {
            std::ifstream is(name, std::ios::binary);
            _p.readFromStream(cs._cs, prog._prog, is);
        }
        
        void activate(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs);

        void learn(PyComputeSystem &cs, const PyIntBuffer &targetCs);

        void save(PyComputeSystem &cs, const std::string &name) {
            std::ofstream os(name, std::ios::binary);
            _p.writeToStream(cs._cs, os);
        }

        int getNumVisibleLayers() const {
            return _p.getNumVisibleLayers();
        }

        const PyPVisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        PyIntBuffer getHiddenCs() const {
            cl_int3 size = _p.getHiddenSize();

            PyIntBuffer buf;
            buf._size = size.x * size.y;
            buf._buf = _p.getHiddenCs();

            return buf;
        }

        PyInt3 getHiddenSize() const {
            cl_int3 size = _p.getHiddenSize();

            return PyInt3(size.x, size.y, size.z);
        }
    };
}