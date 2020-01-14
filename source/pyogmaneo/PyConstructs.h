// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

namespace pyogmaneo {
    class PyInt3 {
    private:
    public:
        int x, y, z;

        PyInt3() 
        :
        x(0),
        y(0),
        z(0)
        {}

        PyInt3(
            int x,
            int y,
            int z
        )
        :
        x(x),
        y(y),
        z(z)
        {}
    };
}