// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyComputeSystem.h"

namespace pyogmaneo {
    class PyIntBuffer {
    private:
        cl::Buffer _buf;

        int _size;

    public:
        PyIntBuffer();

        PyIntBuffer(PyComputeSystem &cs, int size) {
            create(cs, size);
        }

        void create(PyComputeSystem &cs, int size);

        void write(PyComputeSystem &cs, const std::vector<int> &data);
        std::vector<int> read(PyComputeSystem &cs) const;

        int getSize() const {
            return _size;
        }

        friend class PySparseCoder;
        friend class PyPredictor;
        friend class PyActor;
        friend class PyHierarchy;
    };
}