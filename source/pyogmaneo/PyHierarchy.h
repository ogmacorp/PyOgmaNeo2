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
    struct PyLayerDesc {
        PyInt2 _hiddenSize;

        int _scRadius;
        int _pRadius;

        int _ticksPerUpdate;
        int _temporalHorizon;

        PyLayerDesc()
        : _hiddenSize(4, 4), _scRadius(4), _pRadius(4), _ticksPerUpdate(2), _temporalHorizon(2)
        {}

        PyLayerDesc(const PyInt2 &hiddenSize, int scRadius, int pRadius, int ticksPerUpdate, int temporalHorizon)
        : _hiddenSize(hiddenSize), _scRadius(scRadius), _pRadius(pRadius), _ticksPerUpdate(ticksPerUpdate), _temporalHorizon(temporalHorizon)
        {}
    };

    class PyHierarchy {
    private:
        ogmaneo::Hierarchy _h;

    public:
        PyHierarchy(PyComputeSystem &cs, const std::vector<PyInt2> &inputSizes, const std::vector<PyLayerDesc> &layerDescs);
        PyHierarchy(const std::string &fileName);

        void step(PyComputeSystem &cs, const std::vector<std::vector<float> > &inputCs, bool learn = true);

        void save(const std::string &fileName) const;

        int getNumLayers() const {
            return _h.getNumLayers();
        }

        const std::vector<float> &getPredictions(int i) const {
            return _h.getPredictions(i);
        }

        bool getUpdate(int l) const {
            return _h.getUpdate(l);
        }

        const std::vector<float> &getHiddenStates(int l) {
            return _h.getSCLayer(l).getHiddenStates();
        }

        const std::vector<float> &getHiddenBlurs(int l) {
            return _h.getSCLayer(l).getHiddenBlurs();
        }

        const std::vector<float> &getHiddenPredictions(int l) {
            return _h.getSCLayer(l).getHiddenPredictions();
        }

        int getTicks(int l) const {
            return _h.getTicks(l);
        }

        int getTicksPerUpdate(int l) const {
            return _h.getTicksPerUpdate(l);
        }

        void setAlpha(int l, float alpha) {
            _h.getSCLayer(l)._alpha = alpha;
        }

        float getAlpha(int l) const {
            return _h.getSCLayer(l)._alpha;
        }

        void setBeta(int l, float beta) {
            _h.getSCLayer(l)._beta = beta;
        }

        float getBeta(int l) const {
            return _h.getSCLayer(l)._beta;
        }

        void setInhibitRadius(int l, int inhibitRadius) {
            _h.getSCLayer(l)._inhibitRadius = inhibitRadius;
        }

        int getInhibitRadius(int l) const {
            return _h.getSCLayer(l)._inhibitRadius;
        }

        void setBlurRadius(int l, int blurRadius) {
            _h.getSCLayer(l)._blurRadius = blurRadius;
        }

        float getBlurRadius(int l) const {
            return _h.getSCLayer(l)._blurRadius;
        }

        void setRefractoryTime(int l, int refractoryTime) {
            _h.getSCLayer(l)._refractoryTime = refractoryTime;
        }

        int getRefractoryTime(int l) const {
            return _h.getSCLayer(l)._refractoryTime;
        }
    };
}