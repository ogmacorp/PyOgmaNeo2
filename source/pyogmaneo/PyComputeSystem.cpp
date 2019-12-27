// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyComputeSystem.h"

using namespace pyogmaneo;

PyComputeSystem::PyComputeSystem(
    size_t numWorkers,
    unsigned long seed
)
:
_cs(numWorkers)
{
    _cs._rng.seed(seed);
}