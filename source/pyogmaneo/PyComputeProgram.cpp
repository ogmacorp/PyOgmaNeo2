// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyComputeProgram.h"

using namespace pyogmaneo;

PyComputeProgram::PyComputeProgram(PyComputeSystem &cs, const std::string &name) {
    _prog.loadFromFile(cs._cs, name);
}