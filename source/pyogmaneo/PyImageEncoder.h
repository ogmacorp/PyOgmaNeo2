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
#include <ogmaneo/ImageEncoder.h>
#include <fstream>

namespace pyogmaneo {
    struct PyImVisibleLayerDesc {
        PyInt3 _size;

        int _radius;

        PyImVisibleLayerDesc()
        :
        _size(8, 8, 16),
        _radius(2)
        {}

        PyImVisibleLayerDesc(
            const PyInt3 &size,
            int radius)
        : 
        _size(size),
        _radius(radius)
        {}
    };

    class PyImageEncoder {
    private:
        ogmaneo::ImageEncoder _enc;

    public:
        float _alpha;
        float _gamma;
        
        PyImageEncoder(
            PyComputeSystem &cs,
            const PyInt3 &hiddenSize,
            const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs
        );

        PyImageEncoder(
            const std::string &fileName
        );

        void step(
            PyComputeSystem &cs,
            const std::vector<std::vector<float> > &visibleActivations,
            bool learnEnabled = true
        );

        void reconstruct(
            PyComputeSystem &cs,
            const std::vector<int> &hiddenCs
        );

        void save(
            const std::string &fileName
        ) const;

        int getNumVisibleLayers() const {
            return _enc.getNumVisibleLayers();
        }

        const std::vector<float> &getReconstruction(
            int i
        ) const {
            return _enc.getVisibleLayer(i)._reconActs;
        }

        std::vector<int> getHiddenCs() const {
            return _enc.getHiddenCs();
        }

        PyInt3 getHiddenSize() const {
            ogmaneo::Int3 size = _enc.getHiddenSize();

            return PyInt3(size.x, size.y, size.z);
        }

        std::vector<float> getReceptiveField(
            PyComputeSystem &cs,
            int i,
            const PyInt3 &hiddenPosition,
            PyInt3 &size
        ) const;
    };
}