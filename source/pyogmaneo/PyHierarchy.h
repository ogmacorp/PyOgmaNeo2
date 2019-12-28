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
    const int _inputTypePrediction = 1;
    const int _inputTypeAction = 2;

    struct PyLayerDesc {
        PyInt3 _hiddenSize;

        int _ffRadius;
        int _pRadius;

        int _ticksPerUpdate;
        int _temporalHorizon;

        int _aRadius;
        int _historyCapacity;

        PyLayerDesc()
        :
        _hiddenSize(4, 4, 16),
        _ffRadius(2),
        _pRadius(2),
        _ticksPerUpdate(2),
        _temporalHorizon(2),
        _aRadius(2),
        _historyCapacity(16)
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
        _hiddenSize(hiddenSize),
        _ffRadius(ffRadius),
        _pRadius(pRadius),
        _ticksPerUpdate(ticksPerUpdate),
        _temporalHorizon(temporalHorizon),
        _aRadius(aRadius),
        _historyCapacity(historyCapacity)
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
            bool learnEnabled = true,
            float reward = 0.0f
        );

        void save(
            const std::string &fileName
        ) const;

        int getNumLayers() const {
            return _h.getNumLayers();
        }

        const std::vector<int> &getPredictionCs(
            int i
        ) const {
            return _h.getPredictionCs(i);
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

        bool pLayerExists(
            int l,
            int v
        ) {
            return _h.getPLayers(l)[v] != nullptr;
        }

        bool aLayerExists(
            int v
        ) {
            return _h.getALayers()[v] != nullptr;
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
            int v,
            float alpha
        ) {
            assert(_h.getPLayers(l)[v] != nullptr);
            
            _h.getPLayers(l)[v]->_alpha = alpha;
        }

        float getPAlpha(
            int l,
            int v
        ) const {
            assert(_h.getPLayers(l)[v] != nullptr);
            
            return _h.getPLayers(l)[v]->_alpha;
        }

        void setAAlpha(
            int v,
            float alpha
        ) {
            assert(_h.getALayers()[v] != nullptr);
            
            _h.getALayers()[v]->_alpha = alpha;
        }

        float getAAlpha(
            int v
        ) const {
            assert(_h.getALayers()[v] != nullptr);
            
            return _h.getALayers()[v]->_alpha;
        }

        void setABeta(
            int v,
            float beta
        ) {
            assert(_h.getALayers()[v] != nullptr);
            
            _h.getALayers()[v]->_beta = beta;
        }

        float getABeta(
            int v
        ) const {
            assert(_h.getALayers()[v] != nullptr);
            
            return _h.getALayers()[v]->_beta;
        }

        void setAGamma(
            int v,
            float gamma
        ) {
            assert(_h.getALayers()[v] != nullptr);
            
            _h.getALayers()[v]->_gamma = gamma;
        }

        float getAGamma(
            int v
        ) const {
            assert(_h.getALayers()[v] != nullptr);
            
            return _h.getALayers()[v]->_gamma;
        }

        void setAHistoryIters(
            int v,
            int historyIters
        ) {
            assert(_h.getALayers()[v] != nullptr);
            
            _h.getALayers()[v]->_historyIters = historyIters;
        }

        int getAHistoryIters(
            int v
        ) const {
            assert(_h.getALayers()[v] != nullptr);
            
            return _h.getALayers()[v]->_historyIters;
        }

        std::vector<float> getSCReceptiveField(
            PyComputeSystem &cs,
            int l,
            int i,
            const PyInt3 &hiddenPosition,
            PyInt3 &size
        ) const;
    };
}