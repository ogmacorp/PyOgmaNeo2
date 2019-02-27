// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

namespace pyogmaneo {
    class PyInt2 {
    private:
    public:
        int x, y;

        PyInt2() 
        : x(0), y(0)
        {}

        PyInt2(int X, int Y)
        : x(X), y(Y)
        {}
    };
}