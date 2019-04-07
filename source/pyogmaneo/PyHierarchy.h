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

        int _ticksPerUpdate;
        int _temporalHorizon;

        PyLayerDesc()
        : _hiddenSize(4, 4, 16), _scRadius(2), _pRadius(2), _ticksPerUpdate(2), _temporalHorizon(2)
        {}

        PyLayerDesc(const PyInt3 &hiddenSize, int scRadius, int pRadius, int ticksPerUpdate, int temporalHorizon)
        : _hiddenSize(hiddenSize), _scRadius(scRadius), _pRadius(pRadius), _ticksPerUpdate(ticksPerUpdate), _temporalHorizon(temporalHorizon)
        {}
    };

    class PyHierarchy;

    class PyState {
    private:
        ogmaneo::State _state;

    public:
        PyState(const PyHierarchy &h);
        PyState(const std::string &fileName);

        void save(const std::string &fileName) const;

        bool getUpdate(int l) const {
            return _state.getUpdate(l);
        }

        int getTick(int l) const {
            return _state.getTick(l);
        }

        friend class PyHierarchy;
    };

    class PyHierarchy {
    private:
        ogmaneo::Hierarchy _h;

    public:
        PyHierarchy(PyComputeSystem &cs, const std::vector<PyInt3> &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs);
        PyHierarchy(const std::string &fileName);

        void step(PyComputeSystem &cs, const std::vector<std::vector<int> > &inputCs, PyState &state, bool learnEnabled = true);

        void save(const std::string &fileName) const;

        int getNumLayers() const {
            return _h.getNumLayers();
        }

        const std::vector<int> &getPredictionCs(int i) const {
            return _h.getPredictionCs(i);
        }

        const std::vector<int> &getHiddenCs(int l) {
            return _h.getSCLayer(l).getHiddenCs();
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

        void setSCAlpha(int l, float alpha) {
            _h.getSCLayer(l)._alpha = alpha;
        }

        float getSCAlpha(int l) const {
            return _h.getSCLayer(l)._alpha;
        }

        void setSCRefractoryTicks(int l, int refractoryTicks) {
            _h.getSCLayer(l)._refractoryTicks = refractoryTicks;
        }

        int getSCRefractoryTicks(int l) const {
            return _h.getSCLayer(l)._refractoryTicks;
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