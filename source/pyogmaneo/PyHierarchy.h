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
#include <ogmaneo/Hierarchy.h>
#include <fstream>

namespace pyogmaneo {
const int _inputTypeNone = 0;
const int _inputTypeAction = 1;

struct PyLayerDesc {
    PyInt3 _hiddenSize;

    int _ffRadius;
    int _rRadius;

    int _ticksPerUpdate;
    int _temporalHorizon;

    PyLayerDesc()
    :
    _hiddenSize(4, 4, 16),
    _ffRadius(2),
    _rRadius(2),
    _ticksPerUpdate(2),
    _temporalHorizon(2)
    {}

    PyLayerDesc(
        const PyInt3 &hiddenSize,
        int ffRadius,
        int rRadius,
        int ticksPerUpdate,
        int temporalHorizon
    )
    :
    _hiddenSize(hiddenSize),
    _ffRadius(ffRadius),
    _rRadius(rRadius),
    _ticksPerUpdate(ticksPerUpdate),
    _temporalHorizon(temporalHorizon)
    {}
};

class PyHierarchy {
private:
    ogmaneo::Hierarchy _h;

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
        return _h.getNumLayers();
    }

    const std::vector<int> &getActionCs(
        int i
    ) const {
        return _h.getActionCs(i);
    }

    bool getUpdate(
        int l
    ) const {
        return _h.getUpdate(l);
    }

    const std::vector<int> &getHiddenCs(
        int l
    ) {
        return _h.getSCLayer(l).getHiddenCs();
    }

    PyInt3 getHiddenSize(
        int l
    ) {
        ogmaneo::Int3 size = _h.getSCLayer(l).getHiddenSize();

        return { size.x, size.y, size.z };
    }

    int getTicks(
        int l
    ) const {
        return _h.getTicks(l);
    }

    int getTicksPerUpdate(
        int l
    ) const {
        return _h.getTicksPerUpdate(l);
    }

    int getNumSCVisibleLayers(
        int l
    ) {
        return _h.getSCLayer(l).getNumVisibleLayers();
    }

    int getNumInputs() const {
        return _h.getInputSizes().size();
    }

    PyInt3 getInputSize(
        int i
    ) const {
        ogmaneo::Int3 size = _h.getInputSizes()[i];

        return { size.x, size.y, size.z };
    }

    void setSCAlpha(
        int l,
        float alpha
    ) {
        _h.getSCLayer(l)._alpha = alpha;
    }

    float getSCAlpha(
        int l
    ) const {
        return _h.getSCLayer(l)._alpha;
    }

    void setAlpha(
        float alpha
    ) {
        _h._alpha = alpha;
    }

    float getAlpha() const {
        return _h._alpha;
    }

    void setBeta(
        float beta
    ) {
        _h._beta = beta;
    }

    float getBeta() const {
        return _h._beta;
    }

    void setGamma(
        float gamma
    ) {
        _h._gamma = gamma;
    }

    float getGamma() const {
        return _h._gamma;
    }

    void setMaxHistorySamples(
        int maxHistorySamples
    ) {
        _h._maxHistorySamples = maxHistorySamples;
    }

    int setMaxHistorySamples() const {
        return _h._maxHistorySamples;
    }

    void setHistoryIters(
        int historyIters
    ) {
        _h._historyIters = historyIters;
    }

    int getHistoryIters() const {
        return _h._historyIters;
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