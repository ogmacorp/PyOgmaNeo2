// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyComputeProgram.h"
#include "PyIntBuffer.h"
#include "PyFloatBuffer.h"
#include <ogmaneo/neo/ImageEncoder.h>

namespace pyogmaneo {
    struct PyImVisibleLayerDesc {
        std::tuple<int, int, int> _size;

        int _radius;

        PyImVisibleLayerDesc()
        : _size(8, 8, 16), _radius(2)
        {}

        PyImVisibleLayerDesc(std::tuple<int, int, int> size, int radius)
        : _size(size), _radius(radius)
        {}
    };

    class PyImageEncoder {
    private:
        std::vector<PyImVisibleLayerDesc> _visibleLayerDescs;

        ogmaneo::ImageEncoder _enc;

    public:
        float _alpha;
        int _explainIters;

        PyImageEncoder(PyComputeSystem &cs, PyComputeProgram &prog, std::tuple<int, int, int> hiddenSize, const std::vector<PyImVisibleLayerDesc> &visibleLayerDescs);

        void activate(PyComputeSystem &cs, const std::vector<PyFloatBuffer> &visibleAs);

        void learn(PyComputeSystem &cs, const std::vector<PyFloatBuffer> &visibleAs);

        int getNumVisibleLayers() const {
            return _sc.getNumVisibleLayers();
        }

        const PyImVisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        PyIntBuffer getHiddenCs() const {
            PyIntBuffer buf;
            buf._buf = _sc.getHiddenCs();

            return buf;
        }

        std::tuple<int, int, int> getHiddenSize() const {
            cl_int3 size = _sc.getHiddenSize();

            return std::make_tuple(size.x, size.y, size.z);
        }
    };
}