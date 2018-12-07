// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyConstructs.h"
#include "PyComputeSystem.h"
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
        PyImageEncoder(PyComputeSystem &cs, const PyInt3 &hiddenSize, const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs);

        void activate(PyComputeSystem &cs, const std::vector<std::vector<float> > &visibleActivations);

        int getNumVisibleLayers() const {
            return _enc.getNumVisibleLayers();
        }

        const PyImVisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        std::vector<int> getHiddenCs() const {
            return _enc.getHiddenCs();
        }

        PyInt3 getHiddenSize() const {
            ogmaneo::Int3 size = _enc.getHiddenSize();

            return PyInt3(size.x, size.y, size.z);
        }
    };
}