// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyComputeSystem.h"

using namespace pyogmaneo;

PyComputeSystem::PyComputeSystem(const std::string &type) {
    _rng.seed(time(nullptr));

    if (type == "all")
        _cs.create(ogmaneo::ComputeSystem::_all);
    else if (type == "gpu")
        _cs.create(ogmaneo::ComputeSystem::_gpu);
    else if (type == "cpu")
        _cs.create(ogmaneo::ComputeSystem::_cpu);
    else
        std::cerr << "Invalid device type: " << type << std::endl;
}

PyComputeSystem::PyComputeSystem(const std::string &type, unsigned long seed) {
    _rng.seed(seed);

    if (type == "all")
        _cs.create(ogmaneo::ComputeSystem::_all);
    else if (type == "gpu")
        _cs.create(ogmaneo::ComputeSystem::_gpu);
    else if (type == "cpu")
        _cs.create(ogmaneo::ComputeSystem::_cpu);
    else
        std::cerr << "Invalid device type: " << type << std::endl;
}