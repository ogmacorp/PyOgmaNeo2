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
    const int _inputTypePred = 1;

    struct PyLayerDesc {
        PyInt3 _hiddenSize;

        int _scRadius;
        int _pRadius;

        int _ticksPerUpdate;
        int _temporalHorizon;

        int _historyCapacity;

        PyLayerDesc()
        : _hiddenSize(4, 4, 16), _scRadius(2), _pRadius(2), _ticksPerUpdate(2), _temporalHorizon(2), _historyCapacity(64)
        {}

        PyLayerDesc(const PyInt3 &hiddenSize, int scRadius, int pRadius, int ticksPerUpdate, int temporalHorizon)
        : _hiddenSize(hiddenSize), _scRadius(scRadius), _pRadius(pRadius), _ticksPerUpdate(ticksPerUpdate), _temporalHorizon(temporalHorizon)
        {}
    };

    class PyHierarchy {
    private:
        std::vector<PyInt3> _inputSizes;

        ogmaneo::Hierarchy _h;

    public:
        PyHierarchy(PyComputeSystem &cs, const std::vector<PyInt3> &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs);

        void step(PyComputeSystem &cs, const std::vector<std::vector<int> > &inputCs, const std::vector<int> &goalCs, bool learn = true);

        int getNumLayers() const {
            return _h.getNumLayers();
        }

        const std::vector<int> &getPredictionCs(int i) const {
            return _h.getPredictionCs(i);
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
            return _h.getPLayer(l).size();
        }

        void setSCAlpha(int l, float alpha) {
            _h.getSCLayer(l)._alpha = alpha;
        }

        void setSCExplainIters(int l, int explainIters) {
            _h.getSCLayer(l)._explainIters = explainIters;
        }

        void setPAlpha(int l, int v, float alpha) {
            if (_h.getPLayer(l)[v] != nullptr)
                _h.getPLayer(l)[v]->_alpha = alpha;
        }

        void setPGamma(int l, int v, float gamma) {
            if (_h.getPLayer(l)[v] != nullptr)
                _h.getPLayer(l)[v]->_gamma = gamma;
        }

        void setPHistoryIters(int l, int v, int historyIters) {
            if (_h.getPLayer(l)[v] != nullptr)
                _h.getPLayer(l)[v]->_historyIters = historyIters;
        }
    };
}