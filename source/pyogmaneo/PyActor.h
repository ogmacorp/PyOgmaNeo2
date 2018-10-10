// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyComputeProgram.h"
#include "PyIntBuffer.h"
#include <ogmaneo/neo/Actor.h>

namespace pyogmaneo {
    struct PyAVisibleLayerDesc {
        std::tuple<int, int, int> _size;

        int _radius;

        PyAVisibleLayerDesc()
        : _size(8, 8, 16), _radius(2)
        {}

        PyAVisibleLayerDesc(std::tuple<int, int, int> size, int radius)
        : _size(size), _radius(radius)
        {}
    };

    class PyActor {
    private:
        std::vector<PyAVisibleLayerDesc> _visibleLayerDescs;

        ogmaneo::Actor _a;

    public:
        float _alpha;
        float _gamma;

        PyActor(PyComputeSystem &cs, PyComputeProgram &prog, std::tuple<int, int, int> hiddenSize, const std::vector<PyAVisibleLayerDesc> &visibleLayerDescs);

        void step(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs, const PyIntBuffer &targetCs, float reward, bool learn);

        int getNumVisibleLayers() const {
            return _a.getNumVisibleLayers();
        }

        const PyAVisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        PyIntBuffer getHiddenCs() const {
            PyIntBuffer buf;
            buf._buf = _a.getHiddenCs();

            return buf;
        }

        std::tuple<int, int, int> getHiddenSize() const {
            cl_int3 size = _a.getHiddenSize();

            return std::make_tuple(size.x, size.y, size.z);
        }
    };
}