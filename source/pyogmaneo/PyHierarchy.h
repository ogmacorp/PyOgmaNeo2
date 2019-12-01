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
const int _inputTypeAct = 1;

struct PyFirstLayerDesc {
    PyInt3 _hiddenSize;

    int _ffRadius;
    int _aRadius;

    int _temporalHorizon;

    int _historyCapacity;

    PyFirstLayerDesc()
    :
    _hiddenSize(4, 4, 16),
    _ffRadius(2),
    _aRadius(2),
    _temporalHorizon(2),
    _historyCapacity(32)
    {}

    PyFirstLayerDesc(
        const PyInt3 &hiddenSize,
        int ffRadius,
        int aRadius,
        int temporalHorizon,
        int historyCapacity
    )
    :
    _hiddenSize(hiddenSize),
    _ffRadius(ffRadius),
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
        float reward,
        bool learn = true);

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

    PyIntBuffer getActionCs(
        int a
    ) const {
        PyIntBuffer buf;
        buf._size = _inputSizes[a].x * _inputSizes[a].y;
        buf._buf = _h.getActionCs(a);

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

    int getNumALayers(
        int l
    ) {
        return _h.getALayers().size();
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

    bool aLayerExists(
        int a
    ) {
        return _h.getALayers()[a] != nullptr;
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

    void setAAlpha(
        int a,
        float alpha
    ) {
        assert(_h.getALayers()[a] != nullptr);
        
        _h.getALayers()[a]->_alpha = alpha;
    }

    float getAAlpha(
        int a
    ) const {
        assert(_h.getALayers()[a] != nullptr);
        
        return _h.getALayers()[a]->_alpha;
    }

    void setABeta(
        int a,
        float beta
    ) {
        assert(_h.getALayers()[a] != nullptr);
        
        _h.getALayers()[a]->_beta = beta;
    }

    float getABeta(
        int a
    ) const {
        assert(_h.getALayers()[a] != nullptr);
        
        return _h.getALayers()[a]->_beta;
    }

    void setAGamma(
        int a,
        float gamma
    ) {
        assert(_h.getALayers()[a] != nullptr);
        
        _h.getALayers()[a]->_gamma = gamma;
    }

    float getAGamma(
        int a
    ) const {
        assert(_h.getALayers()[a] != nullptr);
        
        return _h.getALayers()[a]->_gamma;
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