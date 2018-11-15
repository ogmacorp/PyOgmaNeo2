// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyConstructs.h"
#include "PyComputeProgram.h"
#include "PyIntBuffer.h"
#include <ogmaneo/neo/Hierarchy.h>
#include <fstream>

namespace pyogmaneo {
    const int _inputTypeNone = 0;
    const int _inputTypePredict = 1;

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

    class PyHierarchy {
    private:
        std::vector<PyInt3> _inputSizes;

        ogmaneo::Hierarchy _h;

    public:
        PyHierarchy(PyComputeSystem &cs, PyComputeProgram &prog, const std::vector<PyInt3> &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs);
        PyHierarchy(PyComputeSystem &cs, PyComputeProgram &prog, const std::string &name);

        void step(PyComputeSystem &cs, const std::vector<PyIntBuffer> &inputCs, bool learn = true);

        void save(PyComputeSystem &cs, const std::string &name) {
            std::ofstream os(name, std::ios::binary);
            _h.writeToStream(cs._cs, os);
        }

        int getNumLayers() const {
            return _h.getNumLayers();
        }

        PyIntBuffer getPredictionCs(int i) const {
            PyIntBuffer buf;
            buf._size = _inputSizes[i].x * _inputSizes[i].y;
            buf._buf = _h.getPredictionCs(i);

            return buf;
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
    };
}