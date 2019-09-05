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

        int _rfRadius; // Revervior forward radius
        int _rrRadius; // Reservior recurrent radius
        int _pRadius; // Predictor radius

        float _rfScale;
        float _rfDropRatio;
        float _rrScale;
        float _rrDropRatio;
        float _rbScale;

        float _pScale;
        float _pDropRatio;

        PyLayerDesc()
        : _hiddenSize(4, 4, 16), _rfRadius(2), _rrRadius(2), _pRadius(2), _rfScale(4.0f), _rfDropRatio(0.0f), _rrScale(2.0f), _rrDropRatio(0.5f), _rbScale(2.0f), _pScale(4.0f), _pDropRatio(0.0f)
        {}

        PyLayerDesc(const PyInt3 &hiddenSize, int rfRadius, int rrRadius, int pRadius, float rfScale, float rfDropRatio, float rrScale, float rrDropRatio, float rbScale, float pScale, float pDropRatio)
        : _hiddenSize(hiddenSize), _rfRadius(rfRadius), _rrRadius(rrRadius), _pRadius(pRadius), _rfScale(rfScale), _rfDropRatio(rfDropRatio), _rrScale(rrScale), _rrDropRatio(rrDropRatio), _rbScale(rbScale), _pScale(pScale), _pDropRatio(pDropRatio)
        {}
    };

    class PyHierarchy {
    private:
        ogmaneo::Hierarchy _h;

    public:
        PyHierarchy(PyComputeSystem &cs, const std::vector<PyInt3> &inputSizes, const std::vector<int> &inputTypes, const std::vector<PyLayerDesc> &layerDescs);
        PyHierarchy(const std::string &fileName);

        void step(PyComputeSystem &cs, const std::vector<std::vector<float> > &inputStates, const std::vector<float> &goalStates, bool learn = true);

        void save(const std::string &fileName) const;

        int getNumLayers() const {
            return _h.getNumLayers();
        }

        const std::vector<float> &getHiddenStates(int i) const {
            return _h.getRLayer(i).getHiddenStates();
        }

        const std::vector<float> &getPredictionStates(int i) const {
            return _h.getPredictionStates(i);
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