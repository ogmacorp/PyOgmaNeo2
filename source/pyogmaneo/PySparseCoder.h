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
#include <ogmaneo/neo/SparseCoder.h>
#include <fstream>

namespace pyogmaneo {
    struct PySCVisibleLayerDesc {
        PyInt3 _size;

        int _radius;

        PySCVisibleLayerDesc()
        : _size(8, 8, 16), _radius(2)
        {}

        PySCVisibleLayerDesc(const PyInt3 &size, int radius)
        : _size(size), _radius(radius)
        {}
    };

    class PySparseCoder {
    private:
        std::vector<PySCVisibleLayerDesc> _visibleLayerDescs;

        ogmaneo::SparseCoder _sc;

    public:
        float _alpha;
        int _explainIters;

        PySparseCoder(PyComputeSystem &cs, PyComputeProgram &prog, const PyInt3 &hiddenSize, const std::vector<PySCVisibleLayerDesc> &visibleLayerDescs);
        PySparseCoder(PyComputeSystem &cs, PyComputeProgram &prog, const std::string &name) {
            std::ifstream is(name, std::ios::binary);
            _sc.readFromStream(cs._cs, prog._prog, is);

            _alpha = _sc._alpha;
            _explainIters = _sc._explainIters;
        }

        void activate(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs);

        void learn(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs);

        void save(PyComputeSystem &cs, const std::string &name) {
            std::ofstream os(name, std::ios::binary);
            _sc.writeToStream(cs._cs, os);
        }

        int getNumVisibleLayers() const {
            return _sc.getNumVisibleLayers();
        }

        const PySCVisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        PyIntBuffer getHiddenCs() const {
            cl_int3 size = _sc.getHiddenSize();

            PyIntBuffer buf;
            buf._size = size.x * size.y;
            buf._buf = _sc.getHiddenCs();

            return buf;
        }

        PyInt3 getHiddenSize() const {
            cl_int3 size = _sc.getHiddenSize();

            return PyInt3(size.x, size.y, size.z);
        }
    };
}