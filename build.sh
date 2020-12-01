#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION.

# rmm build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="clean librmm rmm -v -g -n -s --ptds -h"
HELP="$0 [clean] [librmm] [rmm] [-v] [-g] [-n] [-s] [--ptds] [-h]
   clean  - remove all existing build artifacts and configuration (start over)
   librmm - build and install the librmm C++ code
   rmm    - build and install the rmm Python package
   -v     - verbose build mode
   -g     - build for debug
   -n     - no install step
   -s     - statically link against cudart
   --ptds - enable per-thread default stream
   -h     - print this text

   default action (no args) is to build and install 'librmm' and 'rmm' targets
"
LIBRMM_BUILD_DIR=${LIBRMM_BUILD_DIR:=${REPODIR}/build}
RMM_BUILD_DIR=${REPODIR}/python/build
BUILD_DIRS="${LIBRMM_BUILD_DIR} ${RMM_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_TYPE=Release
INSTALL_TARGET=install
CUDA_STATIC_RUNTIME=OFF
PER_THREAD_DEFAULT_STREAM=OFF
RAN_CMAKE=0

# Set defaults for vars that may not have been defined externally
# If INSTALL_PREFIX is not set, check PREFIX, then check
# CONDA_PREFIX, then fall back to install inside of $LIBRMM_BUILD_DIR
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX:=$LIBRMM_BUILD_DIR/install}}}
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Runs cmake if it has not been run already. Bash work directory
# is always build directory after calling this function
function ensureCMakeRan {
    mkdir -p "${LIBRMM_BUILD_DIR}"
    cd "${LIBRMM_BUILD_DIR}"
    if (( RAN_CMAKE == 0 )); then
        echo "Executing cmake for librmm..."
        cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
              -DCUDA_STATIC_RUNTIME="${CUDA_STATIC_RUNTIME}" \
              -DPER_THREAD_DEFAULT_STREAM="${PER_THREAD_DEFAULT_STREAM}" \
              -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..
        RAN_CMAKE=1
    fi
}

if hasArg -h; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( NUMARGS != 0 )); then
    for a in ${ARGS}; do
	if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
	    echo "Invalid option: ${a}"
	    exit 1
	fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE=1
    set -x
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg -s; then
    CUDA_STATIC_RUNTIME=ON
fi
if hasArg --ptds; then
    PER_THREAD_DEFAULT_STREAM=ON
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d "${bd}" ]; then
            find "${bd}" -mindepth 1 -delete
            rmdir "${bd}" || true
	fi
    done
fi

################################################################################
# Configure, build, and install librmm
if (( NUMARGS == 0 )) || hasArg librmm; then
    ensureCMakeRan
    echo "building librmm..."
    make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        echo "installing librmm..."
        make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} install
    fi
fi

# Build and install the rmm Python package
if (( NUMARGS == 0 )) || hasArg rmm; then
    cd "${REPODIR}/python"
    export INSTALL_PREFIX
    if [[ ${INSTALL_TARGET} != "" ]]; then
        echo "building rmm..."
        python setup.py build_ext --inplace
        echo "installing rmm..."
        python setup.py install --single-version-externally-managed --record=record.txt
    else
        echo "building rmm..."
        python setup.py build_ext --inplace
    fi

fi
