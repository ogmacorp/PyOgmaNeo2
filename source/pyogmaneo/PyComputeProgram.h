// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <ogmaneo/system/ComputeProgram.h>
#include "PyComputeSystem.h"

namespace pyogmaneo {
class PyComputeProgram {
private:
    ogmaneo::ComputeProgram _prog;

public:
    PyComputeProgram(
        PyComputeSystem &cs,
        const std::string &name
    );

    friend class PySparseCoder;
    friend class PyPredictor;
    friend class PyActor;
    friend class PyHierarchy;
};
} // namespace pyogmaneo