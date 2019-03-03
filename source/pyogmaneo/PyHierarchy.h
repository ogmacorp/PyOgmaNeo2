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
    const int _inputTypeAct = 1;

    struct PyLayerDesc {
        PyInt3 _hiddenSize;

        int _scRadius;
        int _aRadius;

        int _ticksPerUpdate;
        int _temporalHorizon;

        int _historyCapacity;

        PyLayerDesc()
        : _hiddenSize(4, 4, 16), _scRadius(2), _aRadius(2), _ticksPerUpdate(2), _temporalHorizon(2), _historyCapacity(128)
        {}

        PyLayerDesc(const PyInt3 &hiddenSize, int scRadius, int aRadius, int ticksPerUpdate, int temporalHorizon, int historyCapacity)
        : _hiddenSize(hiddenSize), _scRadius(scRadius), _aRadius(aRadius), _ticksPerUpdate(ticksPerUpdate), _temporalHorizon(temporalHorizon), _historyCapacity(historyCapacity)
        {}
    };

    class PyHierarchy {
    private:
        ogmaneo::Hierarchy _h;

    public:
        PyHierarchy(PyComputeSystem &cs, const std::vector<PyInt3> &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs);
        PyHierarchy(const std::string &fileName);

        void step(PyComputeSystem &cs, const std::vector<std::vector<int> > &inputCs, float reward, bool learnEnabled = true);

        void save(const std::string &fileName) const;

        int getNumLayers() const {
            return _h.getNumLayers();
        }

        const std::vector<int> &getActionCs(int i) const {
            return _h.getActionCs(i);
        }

        bool getUpdate(int l) const {
            return _h.getUpdate(l);
        }

        PyInt3 getHiddenSize(int l) {
            ogmaneo::Int3 size = _h.getSCLayer(l).getHiddenSize();

            return { size.x, size.y, size.z };
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
            return _h.getALayer(l).size();
        }

        bool visibleLayerExists(int l, int v) {
            return _h.getALayer(l)[v] != nullptr;
        }

        void setSCAlpha(int l, float alpha) {
            _h.getSCLayer(l)._alpha = alpha;
        }

        float getSCAlpha(int l) const {
            return _h.getSCLayer(l)._alpha;
        }

        void setAAlpha(int l, int v, float alpha) {
            assert(_h.getALayer(l)[v] != nullptr);
            
            _h.getALayer(l)[v]->_alpha = alpha;
        }

        float getAAlpha(int l, int v) const {
            assert(_h.getALayer(l)[v] != nullptr);
            
            return _h.getALayer(l)[v]->_alpha;
        }

        void setAGamma(int l, int v, float gamma) {
            assert(_h.getALayer(l)[v] != nullptr);
            
            _h.getALayer(l)[v]->_gamma = gamma;
        }

        float getAGamma(int l, int v) const {
            assert(_h.getALayer(l)[v] != nullptr);
            
            return _h.getALayer(l)[v]->_gamma;
        }

        void setAHistoryIters(int l, int v, int historyIters) {
            assert(_h.getALayer(l)[v] != nullptr);
            
            _h.getALayer(l)[v]->_historyIters = historyIters;
        }

        int getAHistoryIters(int l, int v) const {
            assert(_h.getALayer(l)[v] != nullptr);
            
            return _h.getALayer(l)[v]->_historyIters;
        }
    };
}