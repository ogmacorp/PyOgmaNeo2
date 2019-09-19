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
    const int _inputTypePred = 1;

    struct PyLayerDesc {
        PyInt3 _hiddenSize;

        int _scRadius;
        int _pRadius;
        int _lRadius;

        int _ticksPerUpdate;
        int _temporalHorizon;

        PyLayerDesc()
        : _hiddenSize(4, 4, 16), _scRadius(2), _pRadius(2), _lRadius(2), _ticksPerUpdate(2), _temporalHorizon(2)
        {}

        PyLayerDesc(const PyInt3 &hiddenSize, int scRadius, int pRadius, int lRadius, int ticksPerUpdate, int temporalHorizon)
        : _hiddenSize(hiddenSize), _scRadius(scRadius), _pRadius(pRadius), _lRadius(lRadius), _ticksPerUpdate(ticksPerUpdate), _temporalHorizon(temporalHorizon)
        {}
    };

    class PyHierarchy {
    private:
        ogmaneo::Hierarchy _h;

    public:
        PyHierarchy(PyComputeSystem &cs, const std::vector<PyInt3> &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs);
        PyHierarchy(const std::string &fileName);

        void step(PyComputeSystem &cs, const std::vector<std::vector<int> > &inputCs, bool learn = true);

        void save(const std::string &fileName) const;

        int getNumLayers() const {
            return _h.getNumLayers();
        }

        const std::vector<int> &getPredictionCs(int i) const {
            return _h.getPredictionCs(i);
        }

        bool getUpdate(int l) const {
            return _h.getUpdate(l);
        }

        const std::vector<int> &getHiddenCs(int l) {
            return _h.getSCLayer(l).getHiddenCs();
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

        bool visibleLayerExists(int l, int v) {
            return _h.getPLayer(l)[v] != nullptr;
        }

        void setSCExplainIters(int l, int explainIters) {
            _h.getSCLayer(l)._explainIters = explainIters;
        }

        int getSCExplainIters(int l) const {
            return _h.getSCLayer(l)._explainIters;
        }

        void setPAlpha(int l, int v, float alpha) {
            assert(_h.getPLayer(l)[v] != nullptr);
            
            _h.getPLayer(l)[v]->_alpha = alpha;
        }

        float getPAlpha(int l, int v) const {
            assert(_h.getPLayer(l)[v] != nullptr);
            
            return _h.getPLayer(l)[v]->_alpha;
        }
    };
}