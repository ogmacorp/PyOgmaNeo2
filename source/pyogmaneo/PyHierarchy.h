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
#include <ogmaneo/neo/Hierarchy.h>
#include <fstream>

namespace pyogmaneo {
    const int _inputTypeNone = 0;
    const int _inputTypeAct = 1;

    struct PyLayerDesc {
        PyInt3 _hiddenSize;

        int _scRadius;
        int _aRadius;

        int _ticksPerUpdate;
        int _temporalHorizon;

        int _historyCapacity;

        PyLayerDesc()
        : _hiddenSize(4, 4, 16), _scRadius(2), _aRadius(2), _ticksPerUpdate(2), _temporalHorizon(2), _historyCapacity(64)
        {}

        PyLayerDesc(const PyInt3 &hiddenSize, int scRadius, int aRadius, int ticksPerUpdate, int temporalHorizon)
        : _hiddenSize(hiddenSize), _scRadius(scRadius), _aRadius(aRadius), _ticksPerUpdate(ticksPerUpdate), _temporalHorizon(temporalHorizon)
        {}
    };

    class PyHierarchy {
    private:
        std::vector<PyInt3> _inputSizes;

        ogmaneo::Hierarchy _h;

    public:
        PyHierarchy(PyComputeSystem &cs, const std::vector<PyInt3> &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs);

        void step(PyComputeSystem &cs, const std::vector<std::vector<int> > &inputCs, bool learn = true, float reward = 0.0f);

        int getNumLayers() const {
            return _h.getNumLayers();
        }

        const std::vector<int> &getActionCs(int i) const {
            return _h.getActionCs(i);
        }

        bool getUpdate(int l) const {
            return _h.getUpdate(l);
        }

        int getTicks(int l) const {
            return _h.getTicks(l);
        }

        int getTicksPerUpdate(int l) const {
            return _h.getTicksPerUpdate(l);
        }

        int getNumVisibleLayers(int l) {
            return _h.getALayer(l).size();
        }

        void setSCAlpha(int l, float alpha) {
            _h.getSCLayer(l)._alpha = alpha;
        }

        void setAAlpha(int l, int v, float alpha) {
            if (_h.getALayer(l)[v] != nullptr)
                _h.getALayer(l)[v]->_alpha = alpha;
        }

        void setAGamma(int l, int v, float gamma) {
            if (_h.getALayer(l)[v] != nullptr)
                _h.getALayer(l)[v]->_gamma = gamma;
        }

        void setAEpsilon(int l, int v, float epsilon) {
            if (_h.getALayer(l)[v] != nullptr)
                _h.getALayer(l)[v]->_epsilon = epsilon;
        }

        void setAHistoryIters(int l, int v, int historyIters) {
            if (_h.getALayer(l)[v] != nullptr)
                _h.getALayer(l)[v]->_historyIters = historyIters;
        }
    };
}