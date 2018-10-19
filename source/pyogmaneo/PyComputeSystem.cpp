// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyComputeSystem.h"

using namespace pyogmaneo;

PyComputeSystem::PyComputeSystem(const std::string &type, unsigned long seed, int platformIndex, int deviceIndex) {
    _rng.seed(seed);

    if (type == "all")
        _cs.create(ogmaneo::ComputeSystem::_all, platformIndex, deviceIndex);
    else if (type == "gpu")
        _cs.create(ogmaneo::ComputeSystem::_gpu, platformIndex, deviceIndex);
    else if (type == "cpu")
        _cs.create(ogmaneo::ComputeSystem::_cpu, platformIndex, deviceIndex);
    else
        std::cerr << "Invalid device type: " << type << std::endl;
}