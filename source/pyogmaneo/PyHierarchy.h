// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyConstructs.h"
#include "PyComputeProgram.h"
#include "PyIntBuffer.h"
#include <ogmaneo/Hierarchy.h>
#include <fstream>

namespace pyogmaneo {
const int _inputTypeNone = 0;
const int _inputTypePrediction = 1;
const int _inputTypeAction = 2;

struct PyFirstLayerDesc {
    PyInt3 _hiddenSize;

    int _ffRadius;
    int _pRadius;
    int _aRadius;

    int _temporalHorizon;

    int _historyCapacity;

    PyFirstLayerDesc()
    :
    _hiddenSize(4, 4, 16),
    _ffRadius(2),
    _pRadius(2),
    _aRadius(2),
    _temporalHorizon(2),
    _historyCapacity(32)
    {}

    PyFirstLayerDesc(
        const PyInt3 &hiddenSize,
        int ffRadius,
        int pRadius,
        int aRadius,
        int temporalHorizon,
        int historyCapacity
    )
    :
    _hiddenSize(hiddenSize),
    _ffRadius(ffRadius),
    _pRadius(pRadius),
    _aRadius(aRadius),
    _temporalHorizon(temporalHorizon),
    _historyCapacity(historyCapacity)
    {}
};

struct PyHigherLayerDesc {
    PyInt3 _hiddenSize;

    int _ffRadius;
    int _pRadius;

    int _ticksPerUpdate;
    int _temporalHorizon;

    PyHigherLayerDesc()
    :
    _hiddenSize(4, 4, 16),
    _ffRadius(2),
    _pRadius(2),
    _ticksPerUpdate(2),
    _temporalHorizon(2)
    {}

    PyHigherLayerDesc(
        const PyInt3 &hiddenSize,
        int ffRadius,
        int pRadius,
        int ticksPerUpdate,
        int temporalHorizon
    )
    :
    _hiddenSize(hiddenSize),
    _ffRadius(ffRadius),
    _pRadius(pRadius),
    _ticksPerUpdate(ticksPerUpdate),
    _temporalHorizon(temporalHorizon)
    {}
};

class PyHierarchy {
private:
    std::vector<PyInt3> _inputSizes;

    ogmaneo::Hierarchy _h;

public:
    PyHierarchy(
        PyComputeSystem &cs,
        PyComputeProgram &prog,
        const std::vector<PyInt3> &inputSizes,
        const std::vector<int> &inputTypes,
        const PyFirstLayerDesc &firstLayerDesc,
        const std::vector<PyHigherLayerDesc> &higherLayerDescs
    );

    PyHierarchy(
        PyComputeSystem &cs,
        PyComputeProgram &prog,
        const std::string &name
    );

    void step(
        PyComputeSystem &cs,
        const std::vector<PyIntBuffer> &inputCs,
        bool learnEnabled = true,
        float reward = 0.0f);

    void save(
        PyComputeSystem &cs,
        const std::string &name
    ) {
        std::ofstream os(name, std::ios::binary);
        _h.writeToStream(cs._cs, os);
    }

    int getNumLayers() const {
        return _h.getNumLayers();
    }

    PyIntBuffer getPredictionCs(
        int i
    ) const {
        PyIntBuffer buf;
        buf._size = _inputSizes[i].x * _inputSizes[i].y;
        buf._buf = _h.getPredictionCs(i);

        return buf;
    }

    PyIntBuffer getActionCs(
        int i
    ) const {
        PyIntBuffer buf;
        buf._size = _inputSizes[i].x * _inputSizes[i].y;
        buf._buf = _h.getActionCs(i);

        return buf;
    }

    bool getUpdate(
        int l
    ) const {
        return _h.getUpdate(l);
    }

    PyIntBuffer getHiddenCs(
        int l
    ) {
        PyIntBuffer buf;
        buf._size = _h.getSCLayer(l).getHiddenSize().x * _h.getSCLayer(l).getHiddenSize().y;
        buf._buf = _h.getSCLayer(l).getHiddenCs();

        return buf;
    }

    PyInt3 getHiddenSize(
        int l
    ) {
        return PyInt3(_h.getSCLayer(l).getHiddenSize().x, _h.getSCLayer(l).getHiddenSize().y, _h.getSCLayer(l).getHiddenSize().z);
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

    int getNumInputLayers(
        int l
    ) {
        return _h.getInputSizes().size();
    }

    int getNumPLayers(
        int l
    ) {
        return _h.getPLayers(l).size();
    }

    int getNumSCVisibleLayers(
        int l
    ) const {
        return _h.getSCLayer(l).getNumVisibleLayers();
    }

    bool pLayerExists(
        int i
    ) {
        return _h.getBPLayers()[i] != nullptr;
    }

    bool aLayerExists(
        int i
    ) {
        return _h.getBALayers()[i] != nullptr;
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

    void setPAlpha(
        int l,
        int p,
        float alpha
    ) {
        _h.getPLayers(l)[p]._alpha = alpha;
    }

    float getPAlpha(
        int l,
        int p
    ) const {
        return _h.getPLayers(l)[p]._alpha;
    }

    void setBPAlpha(
        int i,
        float alpha
    ) {
        assert(_h.getBPLayers()[i] != nullptr);
        
        _h.getBPLayers()[i]->_alpha = alpha;
    }

    float getBPAlpha(
        int i
    ) const {
        assert(_h.getBPLayers()[i] != nullptr);
        
        return _h.getBPLayers()[i]->_alpha;
    }

    void setBAAlpha(
        int i,
        float alpha
    ) {
        assert(_h.getBALayers()[i] != nullptr);
        
        _h.getBALayers()[i]->_alpha = alpha;
    }

    float getBAAlpha(
        int i
    ) const {
        assert(_h.getBALayers()[i] != nullptr);
        
        return _h.getBALayers()[i]->_alpha;
    }

    void setBABeta(
        int i,
        float beta
    ) {
        assert(_h.getALayers()[i] != nullptr);
        
        _h.getBALayers()[i]->_beta = beta;
    }

    float getBABeta(
        int i
    ) const {
        assert(_h.getBALayers()[i] != nullptr);
        
        return _h.getBALayers()[i]->_beta;
    }

    void setBAGamma(
        int i,
        float gamma
    ) {
        assert(_h.getBALayers()[i] != nullptr);
        
        _h.getBALayers()[i]->_gamma = gamma;
    }

    float getBAGamma(
        int i
    ) const {
        assert(_h.getBALayers()[i] != nullptr);
        
        return _h.getBALayers()[i]->_gamma;
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