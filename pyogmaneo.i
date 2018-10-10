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

%include "std_pair.i"
%include "std_string.i"
%include "std_vector.i"

%{
#include "PyComputeSystem.h"
#include "PyComputeProgram.h"
#include "PySparseCoder.h"
#include "PyPredictor.h"
#include "PyActor.h"
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
%template(StdInt3) std::tuple<int, int, int>;
%template(StdVecInt3) std::vector<std::tuple<int, int, int> >;

%include "PyComputeSystem.h"
%include "PyComputeProgram.h"
%include "PySparseCoder.h"
%include "PyPredictor.h"
%include "PyActor.h"
%include "PyHierarchy.h"
%include "PyImageEncoder.h"
