// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyOgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

%begin %{
#include <cmath>
#include <iostream>
%}
%module pyogmaneo

%include "std_array.i"
%include "std_string.i"
%include "std_vector.i"

%{
#include "PyConstructs.h"
#include "PyComputeSystem.h"
#include "PyHierarchy.h"
%}

// Handle STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%template(StdVeci) std::vector<int>;
%template(StdVec2Di) std::vector<std::vector<int> >;
%template(StdVecf) std::vector<float>;
%template(StdVec2Df) std::vector<std::vector<float> >;
%template(StdVecInt3) std::vector<pyogmaneo::PyInt3>;
%template(StdVecLayerDesc) std::vector<pyogmaneo::PyLayerDesc>;

%include "PyConstructs.h"
%include "PyComputeSystem.h"
%include "PyHierarchy.h"