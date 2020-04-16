// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyComputeSystem.h"

using namespace pyogmaneo;

PyComputeSystem::PyComputeSystem(
    unsigned long seed
) {
    cs.rng.seed(seed);
}

void PyComputeSystem::setNumThreads(int numThreads) {
    ogmaneo::ComputeSystem::setNumThreads(numThreads);
}

int PyComputeSystem::getNumThreads() {
    return ogmaneo::ComputeSystem::getNumThreads();
}