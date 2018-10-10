// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyComputeProgram.h"
#include "PyIntBuffer.h"
#include <ogmaneo/neo/SparseCoder.h>

namespace pyogmaneo {
    struct PySCVisibleLayerDesc {
        std::array<int, 3> _size;

        int _radius;

        PySCVisibleLayerDesc()
        : _size({ 8, 8, 16 }), _radius(2)
        {}

        PySCVisibleLayerDesc(std::array<int, 3> size, int radius)
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

        PySparseCoder(PyComputeSystem &cs, PyComputeProgram &prog, std::array<int, 3> hiddenSize, const std::vector<PySCVisibleLayerDesc> &visibleLayerDescs);

        void activate(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs);

        void learn(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs);

        int getNumVisibleLayers() const {
            return _sc.getNumVisibleLayers();
        }

        const PySCVisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        PyIntBuffer getHiddenCs() const {
            PyIntBuffer buf;
            buf._buf = _sc.getHiddenCs();

            return buf;
        }

        std::array<int, 3> getHiddenSize() const {
            cl_int3 size = _sc.getHiddenSize();

            return { size.x, size.y, size.z };
        }
    };
}