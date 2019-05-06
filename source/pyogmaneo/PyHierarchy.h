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

        int _mRadius;
        int _cRadius;

        int _ticksPerUpdate;
        int _temporalHorizon;

        PyLayerDesc()
        : _hiddenSize(4, 4), _mRadius(4), _cRadius(4), _ticksPerUpdate(2), _temporalHorizon(2)
        {}

        PyLayerDesc(const PyInt2 &hiddenSize, int mRadius, int cRadius, int ticksPerUpdate, int temporalHorizon)
        : _hiddenSize(hiddenSize), _mRadius(mRadius), _cRadius(cRadius), _ticksPerUpdate(ticksPerUpdate), _temporalHorizon(temporalHorizon)
        {}
    };

    class PyHierarchy {
    private:
        ogmaneo::Hierarchy _h;

    public:
        PyHierarchy(PyComputeSystem &cs, const std::vector<PyInt2> &inputSizes, const std::vector<PyLayerDesc> &layerDescs);
        PyHierarchy(const std::string &fileName);

        void step(PyComputeSystem &cs, const std::vector<std::vector<float> > &inputCs, const std::vector<float> &topFeedBackStates, bool learn = true);

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
            return _h.getLayer(l).getHiddenStates();
        }

        const std::vector<float> &getHiddenBlurs(int l) {
            return _h.getLayer(l).getHiddenBlurs();
        }

        const std::vector<float> &getHiddenPredictions(int l) {
            return _h.getLayer(l).getHiddenPredictions();
        }

        int getTicks(int l) const {
            return _h.getTicks(l);
        }

        int getTicksPerUpdate(int l) const {
            return _h.getTicksPerUpdate(l);
        }

        void setAlpha(int l, float alpha) {
            _h.getLayer(l)._alpha = alpha;
        }

        float getAlpha(int l) const {
            return _h.getLayer(l)._alpha;
        }

        void setGamma(int l, float gamma) {
            _h.getLayer(l)._gamma = gamma;
        }

        float getGamma(int l) const {
            return _h.getLayer(l)._gamma;
        }

        void setInhibitRadius(int l, int inhibitRadius) {
            _h.getLayer(l)._inhibitRadius = inhibitRadius;
        }

        int getInhibitRadius(int l) const {
            return _h.getLayer(l)._inhibitRadius;
        }

        void setBlurRadius(int l, int blurRadius) {
            _h.getLayer(l)._blurRadius = blurRadius;
        }

        int getBlurRadius(int l) const {
            return _h.getLayer(l)._blurRadius;
        }

        void setPlanRadius(int l, int planRadius) {
            _h.getLayer(l)._planRadius = planRadius;
        }

        int getPlanRadius(int l) const {
            return _h.getLayer(l)._planRadius;
        }

        void setPlanIters(int l, int planIters) {
            _h.getLayer(l)._planIters = planIters;
        }

        int getPlanIters(int l) const {
            return _h.getLayer(l)._planIters;
        }
    };
}