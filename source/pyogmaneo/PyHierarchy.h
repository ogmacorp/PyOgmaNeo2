// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyComputeProgram.h"
#include "PyIntBuffer.h"
#include <ogmaneo/neo/Hierarchy.h>

namespace pyogmaneo {
    const int _inputTypeNone = 0;
    const int _inputTypePredict = 1;
    const int _inputTypeAct = 2;

    struct PyLayerDesc {
        std::array<int, 3> _hiddenSize;

        int _scRadius;
        int _pRadius;

        int _ticksPerUpdate;
        int _temporalHorizon;

        PyLayerDesc()
        : _hiddenSize({ 4, 4, 16 }), _scRadius(2), _pRadius(2), _ticksPerUpdate(2), _temporalHorizon(2)
        {}

        PyLayerDesc(std::array<int, 3> hiddenSize, int scRadius, int pRadius, int ticksPerUpdate, int temporalHorizon)
        : _hiddenSize(hiddenSize), _scRadius(scRadius), _pRadius(pRadius), _ticksPerUpdate(ticksPerUpdate), _temporalHorizon(temporalHorizon)
        {}
    };

    class PyHierarchy {
    private:
        std::vector<PyLayerDesc> _layerDescs;

        ogmaneo::Hierarchy _h;

    public:
        PyHierarchy(PyComputeSystem &cs, PyComputeProgram &prog, const std::vector<std::array<int, 3> > &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs);

        void step(PyComputeSystem &cs, const std::vector<PyIntBuffer> &inputCs, const PyIntBuffer &topFeedBack, bool learn = true, float reward = 0.0f);

        int getNumLayers() const {
            return _h.getNumLayers();
        }

        PyIntBuffer getPredictionCs(int i) const {
            PyIntBuffer buf;
            buf._buf = _h.getPredictionCs(i);

            return buf;
        }

        PyIntBuffer getActionCs(int i) const {
            PyIntBuffer buf;
            buf._buf = _h.getActionCs(i);

            return buf;
        }

        /*!
        \brief Whether this layer received on update this timestep
        */
        bool getUpdate(int l) const {
            return _h.getUpdate(l);
        }

        /*!
        \brief Get current layer ticks, relative to previous layer
        */
        int getTicks(int l) const {
            return _h.getTicks(l);
        }

        /*!
        \brief Get layer ticks per update, relative to previous layer
        */
        int getTicksPerUpdate(int l) const {
            return _h.getTicksPerUpdate(l);
        }
    };
}