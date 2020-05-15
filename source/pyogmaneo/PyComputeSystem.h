// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
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
    ogmaneo::ComputeSystem cs;

public:
    PyComputeSystem(
        unsigned long seed = 1234
    );

    static void setNumThreads(
        int numThreads
    );
    
    static int getNumThreads();

    friend class PyHierarchy;
    friend class PyImageEncoder;
};
} // namespace pyogmaneo