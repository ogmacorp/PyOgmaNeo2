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
#include <ogmaneo/Hierarchy.h>
#include <fstream>

namespace pyogmaneo {
const int inputTypeNone = 0;
const int inputTypeAction = 1;

struct PyLayerDesc {
    PyInt3 hiddenSize;

    int ffRadius;
    int rRadius;

    int ticksPerUpdate;
    int temporalHorizon;

    PyLayerDesc()
    :
    hiddenSize(4, 4, 16),
    ffRadius(2),
    rRadius(2),
    ticksPerUpdate(2),
    temporalHorizon(2)
    {}

    PyLayerDesc(
        const PyInt3 &hiddenSize,
        int ffRadius,
        int rRadius,
        int ticksPerUpdate,
        int temporalHorizon
    )
    :
    hiddenSize(hiddenSize),
    ffRadius(ffRadius),
    rRadius(rRadius),
    ticksPerUpdate(ticksPerUpdate),
    temporalHorizon(temporalHorizon)
    {}
};

class PyHierarchy {
private:
    ogmaneo::Hierarchy h;

public:
    PyHierarchy(
        PyComputeSystem &cs,
        const std::vector<PyInt3> &inputSizes,
        const std::vector<int> &inputTypes,
        const std::vector<PyLayerDesc> &layerDescs
    );

    PyHierarchy(
        const std::string &fileName
    );

    void step(
        PyComputeSystem &cs,
        const std::vector<std::vector<int> > &inputCs,
        float reward,
        bool learnEnabled = true
    );

    void save(
        const std::string &fileName
    ) const;

    int getNumLayers() const {
        return h.getNumLayers();
    }

    const std::vector<int> &getActionCs(
        int i
    ) const {
        return h.getActionCs(i);
    }

    bool getUpdate(
        int l
    ) const {
        return h.getUpdate(l);
    }

    const std::vector<int> &getHiddenCs(
        int l
    ) {
        return h.getSCLayer(l).getHiddenCs();
    }

    PyInt3 getHiddenSize(
        int l
    ) {
        ogmaneo::Int3 size = h.getSCLayer(l).getHiddenSize();

        return { size.x, size.y, size.z };
    }

    int getTicks(
        int l
    ) const {
        return h.getTicks(l);
    }

    int getTicksPerUpdate(
        int l
    ) const {
        return h.getTicksPerUpdate(l);
    }

    int getNumSCVisibleLayers(
        int l
    ) {
        return h.getSCLayer(l).getNumVisibleLayers();
    }

    int getNumInputs() const {
        return h.getInputSizes().size();
    }

    PyInt3 getInputSize(
        int i
    ) const {
        ogmaneo::Int3 size = h.getInputSizes()[i];

        return { size.x, size.y, size.z };
    }

    void setSCScale(
        int l,
        float scale
    ) {
        h.getSCLayer(l).scale = scale;
    }

    float getSCScale(
        int l
    ) const {
        return h.getSCLayer(l).scale;
    }

    void setAlpha(
        float alpha
    ) {
        h.alpha = alpha;
    }

    float getAlpha() const {
        return h.alpha;
    }

    void setBeta(
        float beta
    ) {
        h.beta = beta;
    }

    float getBeta() const {
        return h.beta;
    }

    void setGamma(
        float gamma
    ) {
        h.gamma = gamma;
    }

    float getGamma() const {
        return h.gamma;
    }

    void setMaxHistorySamples(
        int maxHistorySamples
    ) {
        h.maxHistorySamples = maxHistorySamples;
    }

    int setMaxHistorySamples() const {
        return h.maxHistorySamples;
    }

    void setHistoryIters(
        int historyIters
    ) {
        h.historyIters = historyIters;
    }

    int getHistoryIters() const {
        return h.historyIters;
    }

    std::vector<float> getSCReceptiveField(
        PyComputeSystem &cs,
        int l,
        int i,
        const PyInt3 &hiddenPosition,
        PyInt3 &size
    ) const;
};
} // namespace pyogmaneo