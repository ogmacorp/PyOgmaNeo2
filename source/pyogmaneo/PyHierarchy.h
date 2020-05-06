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
const int inputTypePrediction = 1;
const int inputTypeAction = 2;

struct PyLayerDesc {
    PyInt3 hiddenSize;

    int ffRadius;
    int pRadius;

    int ticksPerUpdate;
    int temporalHorizon;

    int aRadius;
    int historyCapacity;

    PyLayerDesc()
    :
    hiddenSize(4, 4, 16),
    ffRadius(2),
    pRadius(2),
    ticksPerUpdate(2),
    temporalHorizon(4),
    aRadius(2),
    historyCapacity(32)
    {}

    PyLayerDesc(
        const PyInt3 &hiddenSize,
        int ffRadius,
        int pRadius,
        int ticksPerUpdate,
        int temporalHorizon,
        int aRadius,
        int historyCapacity
    )
    :
    hiddenSize(hiddenSize),
    ffRadius(ffRadius),
    pRadius(pRadius),
    ticksPerUpdate(ticksPerUpdate),
    temporalHorizon(temporalHorizon),
    aRadius(aRadius),
    historyCapacity(historyCapacity)
    {}
};

struct PyState {
private:
    ogmaneo::State state;

    friend class PyHierarchy;
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
        bool learnEnabled = true,
        float reward = 0.0f,
        bool mimic = false
    );

    void getState(
        PyState &state
    ) const;

    void setState(
        const PyState &state
    );

    void save(
        const std::string &fileName
    ) const;

    int getNumLayers() const {
        return h.getNumLayers();
    }

    const std::vector<int> &getPredictionCs(
        int i
    ) const {
        return h.getPredictionCs(i);
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

    bool pLayerExists(
        int l,
        int v
    ) {
        return h.getPLayers(l)[v] != nullptr;
    }

    bool aLayerExists(
        int v
    ) {
        return h.getALayers()[v] != nullptr;
    }

    void setSCAlpha(
        int l,
        float alpha
    ) {
        h.getSCLayer(l).alpha = alpha;
    }

    float getSCAlpha(
        int l
    ) const {
        return h.getSCLayer(l).alpha;
    }

    void setPAlpha(
        int l,
        int v,
        float alpha
    ) {
        assert(h.getPLayers(l)[v] != nullptr);
        
        h.getPLayers(l)[v]->alpha = alpha;
    }

    float getPAlpha(
        int l,
        int v
    ) const {
        assert(h.getPLayers(l)[v] != nullptr);
        
        return h.getPLayers(l)[v]->alpha;
    }

    void setAAlpha(
        int v,
        float alpha
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->alpha = alpha;
    }

    float getAAlpha(
        int v
    ) const {
        assert(h.getALayers()[v] != nullptr);
        
        return h.getALayers()[v]->alpha;
    }

    void setABeta(
        int v,
        float beta
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->beta = beta;
    }

    float getABeta(
        int v
    ) const {
        assert(h.getALayers()[v] != nullptr);
        
        return h.getALayers()[v]->beta;
    }

    void setAGamma(
        int v,
        float gamma
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->gamma = gamma;
    }

    float getAGamma(
        int v
    ) const {
        assert(h.getALayers()[v] != nullptr);
        
        return h.getALayers()[v]->gamma;
    }

    void setAMinSteps(
        int v,
        int minSteps
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->minSteps = minSteps;
    }

    int getAMinSteps(
        int v
    ) const {
        assert(h.getALayers()[v] != nullptr);
        
        return h.getALayers()[v]->minSteps;
    }

    void setAHistoryIters(
        int v,
        int historyIters
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->historyIters = historyIters;
    }

    int getAHistoryIters(
        int v
    ) const {
        assert(h.getALayers()[v] != nullptr);
        
        return h.getALayers()[v]->historyIters;
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