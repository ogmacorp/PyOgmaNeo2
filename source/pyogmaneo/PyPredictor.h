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
#include <ogmaneo/neo/Predictor.h>

namespace pyogmaneo {
    struct PyPVisibleLayerDesc {
        std::array<int, 3> _size;

        int _radius;

        PyPVisibleLayerDesc()
        : _size({ 8, 8, 16 }), _radius(2)
        {}

        PyPVisibleLayerDesc(std::array<int, 3> size, int radius)
        : _size(size), _radius(radius)
        {}
    };

    class PyPredictor {
    private:
        std::vector<PyPVisibleLayerDesc> _visibleLayerDescs;

        ogmaneo::Predictor _p;

    public:
        float _alpha;

        PyPredictor(PyComputeSystem &cs, PyComputeProgram &prog, std::array<int, 3> hiddenSize, const std::vector<PyPVisibleLayerDesc> &visibleLayerDescs);

        void activate(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs);

        void learn(PyComputeSystem &cs, const PyIntBuffer &targetCs);

        int getNumVisibleLayers() const {
            return _p.getNumVisibleLayers();
        }

        const PyPVisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        PyIntBuffer getHiddenCs() const {
            PyIntBuffer buf;
            buf._buf = _p.getHiddenCs();

            return buf;
        }

        std::array<int, 3> getHiddenSize() const {
            cl_int3 size = _p.getHiddenSize();

            return { size.x, size.y, size.z };
        }
    };
}