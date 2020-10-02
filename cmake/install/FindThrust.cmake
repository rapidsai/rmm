##============================================================================
## Copyright (c) 2014,
## Sandia Corporation, Los Alamos National Security, LLC.,
## UT-Battelle, LLC., Kitware Inc., University of California Davis
## All rights reserved.
##
## Sandia National Laboratories, New Mexico
## PO Box 5800
## Albuquerque, NM 87185
## USA
## 
## UT-Battelle
## 1 Bethel Valley Rd
## Oak Ridge, TN 37830
##
## Los Alamos National Security, LLC
## 105 Central Park Square
## Los Alamos, NM 87544
##
## Kitware Inc.
## 28 Corporate Drive
## Clifton Park, NY 12065
## USA
##
## University of California, Davis
## One Shields Avenue
## Davis, CA 95616
## USA
##
## Under the terms of Contract DE-AC04-94AL85000, there is a
## non-exclusive license for use of this work by or on behalf of the
## U.S. Government.
##
## Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
## Laboratory (LANL), the U.S. Government retains certain rights in
## this software.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##
## * Redistributions of source code must retain the above copyright
##   notice, this list of conditions and the following disclaimer.
##
## * Redistributions in binary form must reproduce the above copyright
##   notice, this list of conditions and the following disclaimer in the
##   documentation and/or other materials provided with the
##   distribution.
##
## * Neither the name of Kitware nor the names of any contributors may
##   be used to endorse or promote products derived from this software
##   without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
## NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
## SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##============================================================================

#
# FindThrust
#
# This module finds the Thrust header files and extrats their version.  It
# sets the following variables.
#
# THRUST_INCLUDE_DIR -  Include directory for thrust header files.  (All header
#                       files will actually be in the thrust subdirectory.)
# THRUST_VERSION -      Version of thrust in the form "major.minor.patch".
#

find_path( THRUST_INCLUDE_DIR
  HINTS
    /usr/include/cuda
    /usr/local/include
    /usr/local/cuda/include
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CUDA_SDK_ROOT_DIR}
  NAMES thrust/version.h
  DOC "Thrust headers"
  )
if( THRUST_INCLUDE_DIR )
  list( REMOVE_DUPLICATES THRUST_INCLUDE_DIR )
endif( THRUST_INCLUDE_DIR )

# Find thrust version
if (THRUST_INCLUDE_DIR)
  file( STRINGS ${THRUST_INCLUDE_DIR}/thrust/version.h
    version
    REGEX "#define THRUST_VERSION[ \t]+([0-9x]+)"
    )
  string( REGEX REPLACE
    "#define THRUST_VERSION[ \t]+"
    ""
    version
    "${version}"
    )

  math(EXPR major "${version} / 100000")
  math(EXPR minor "(${version} / 100) % 1000")
  math(EXPR version "${version} % 100")
  set( THRUST_VERSION "${major}.${minor}.${version}")
  set( THRUST_MAJOR_VERSION "${major}")
  set( THRUST_MINOR_VERSION "${minor}")
endif()

# Check for required components
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Thrust
  REQUIRED_VARS THRUST_INCLUDE_DIR
  VERSION_VAR THRUST_VERSION
  )

set(THRUST_INCLUDE_DIRS ${THRUST_INCLUDE_DIR})
mark_as_advanced(THRUST_INCLUDE_DIR)
