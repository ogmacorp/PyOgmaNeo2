// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
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
#include "PyComputeProgram.h"
#include "PyIntBuffer.h"
#include "PyFloatBuffer.h"
#include "PyHierarchy.h"
#include "PyImageEncoder.h"
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
%template(StdVecf) std::vector<float>;
%template(StdVecInt3) std::vector<pyogmaneo::PyInt3>;
%template(StdVecIntBuffer) std::vector<pyogmaneo::PyIntBuffer>;
%template(StdVecFloatBuffer) std::vector<pyogmaneo::PyFloatBuffer>;
%template(StdVecHigherLayerDesc) std::vector<pyogmaneo::PyHigherLayerDesc>;

%template(StdVecImVisibleLayerDesc) std::vector<pyogmaneo::PyImVisibleLayerDesc>;

%rename("%(strip:[Py])s") ""; // Remove Py prefix that was added to avoid naming collisions

%include "PyConstructs.h"
%include "PyComputeSystem.h"
%include "PyComputeProgram.h"
%include "PyIntBuffer.h"
%include "PyFloatBuffer.h"
%include "PyHierarchy.h"
%include "PyImageEncoder.h"
