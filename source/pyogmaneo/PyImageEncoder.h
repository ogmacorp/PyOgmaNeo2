// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
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
    PyInt3 size;

    int radius;

    PyImVisibleLayerDesc()
    :
    size(8, 8, 16),
    radius(2)
    {}

    PyImVisibleLayerDesc(
        const PyInt3 &size,
        int radius)
    : 
    size(size),
    radius(radius)
    {}
};

class PyImageEncoder {
private:
    ogmaneo::ImageEncoder enc;

public:
    float alpha;
    float gamma;
    
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
        return enc.getNumVisibleLayers();
    }

    const std::vector<float> &getReconstruction(
        int i
    ) const {
        return enc.getVisibleLayer(i).reconstructions;
    }

    std::vector<int> getHiddenCs() const {
        return enc.getHiddenCs();
    }

    PyInt3 getHiddenSize() const {
        ogmaneo::Int3 size = enc.getHiddenSize();

        return PyInt3(size.x, size.y, size.z);
    }

    PyInt3 getVisibleSize(
        int i
    ) const {
        ogmaneo::Int3 size = enc.getVisibleLayerDesc(i).size;

        return PyInt3(size.x, size.y, size.z);
    }

    std::vector<float> getReceptiveField(
        PyComputeSystem &cs,
        int i,
        const PyInt3 &hiddenPosition,
        PyInt3 &size
    ) const;
};
} // namespace pyogmaneo