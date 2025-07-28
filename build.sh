#!/bin/bash

# Copyright (c) 2019-2025, NVIDIA CORPORATION.

# rmm build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details).

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd "$(dirname "$0")"; pwd)

VALIDARGS="clean librmm rmm -v -g -n -s --ptds -h tests benchmarks"
HELP="$0 [clean] [librmm] [rmm] [-v] [-g] [-n] [-s] [--ptds] [--cmake-args=\"<args>\"] [-h]
   clean                       - remove all existing build artifacts and configuration (start over)
   librmm                      - build and install the librmm C++ code
   rmm                         - build and install the rmm Python package
   benchmarks                  - build benchmarks
   tests                       - build tests
   -v                          - verbose build mode
   -g                          - build for debug
   -n                          - no install step (does not affect Python)
   -s                          - statically link against cudart
   --ptds                      - enable per-thread default stream
   --cmake-args=\\\"<args>\\\" - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   -h                          - print this text

   default action (no args) is to build and install 'librmm' and 'rmm' targets
"
LIBRMM_BUILD_DIR=${LIBRMM_BUILD_DIR:=${REPODIR}/cpp/build}
RMM_BUILD_DIR="${REPODIR}/python/rmm/build"
BUILD_DIRS="${LIBRMM_BUILD_DIR} ${RMM_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_BENCHMARKS=OFF
BUILD_TESTS=OFF
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

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo "$ARGS" | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo "$ARGS" | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo "$ARGS" | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo "$EXTRA_CMAKE_ARGS" | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi
    read -ra EXTRA_CMAKE_ARGS <<< "$EXTRA_CMAKE_ARGS"
}


# Runs cmake if it has not been run already for build directory
# LIBRMM_BUILD_DIR
function ensureCMakeRan {
    mkdir -p "${LIBRMM_BUILD_DIR}"
    if (( RAN_CMAKE == 0 )); then
        echo "Executing cmake for librmm..."
        cmake -S "${REPODIR}"/cpp -B "${LIBRMM_BUILD_DIR}" \
              -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
              -DCUDA_STATIC_RUNTIME="${CUDA_STATIC_RUNTIME}" \
              -DPER_THREAD_DEFAULT_STREAM="${PER_THREAD_DEFAULT_STREAM}" \
              -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
              -DBUILD_TESTS=${BUILD_TESTS} \
              -DBUILD_BENCHMARKS=${BUILD_BENCHMARKS} \
              "${EXTRA_CMAKE_ARGS[@]}"
        RAN_CMAKE=1
    fi
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( NUMARGS != 0 )); then
    # Check for cmake args
    cmakeArgs
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option or formatting, check --help: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG=-v
    set -x
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg benchmarks; then
    BUILD_BENCHMARKS=ON
fi
if hasArg tests; then
    BUILD_TESTS=ON
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
    cmake --build "${LIBRMM_BUILD_DIR}" -j"${PARALLEL_LEVEL}" ${VERBOSE_FLAG}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        echo "installing librmm..."
        cmake --build "${LIBRMM_BUILD_DIR}" --target install -v ${VERBOSE_FLAG}
    fi
fi

# Build and install the rmm Python package
if (( NUMARGS == 0 )) || hasArg rmm; then
    echo "building and installing rmm..."
    SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX};$(IFS=';'; echo "${EXTRA_CMAKE_ARGS[*]}")" python -m pip install \
        --no-build-isolation \
        --no-deps \
        --config-settings rapidsai.disable-cuda=true \
        "${REPODIR}"/python/rmm
fi
