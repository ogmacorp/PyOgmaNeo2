// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <ogmaneo/ComputeSystem.h>
#include <random>
#include <iostream>

namespace pyogmaneo {
class PyComputeSystem {
private:
    ogmaneo::ComputeSystem _cs;
    std::mt19937 _rng;

public:
    PyComputeSystem(
        const std::string &type = "gpu",
        unsigned long seed = 1234,
        int platformIndex = -1,
        int deviceIndex = -1
    );

    friend class PyComputeProgram;
    friend class PyIntBuffer;
    friend class PyFloatBuffer;

    friend class PySparseCoder;
    friend class PyPredictor;
    friend class PyActor;
    friend class PyHierarchy;
};
} // namespace pyogmaneo