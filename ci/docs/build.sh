#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# rmm GPU build & testscript for CI  #
######################################
set -ex

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export HOME=$WORKSPACE
export DOCS_DIR=/data/docs/html

while getopts "d" option; do
    case ${option} in
        d)
            DOCS_DIR=${OPTARG}
            ;;
    esac
done

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf
conda install -c conda-forge doxygen

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

################################################################################
# BUILD - Build librmm
################################################################################

#logger "Build librmm..."
CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX11_ABI=$BUILD_ABI"

# Use CMake-based build procedure
mkdir -p $WORKSPACE/build
cd build
# configure
cmake $CMAKE_COMMON_VARIABLES ..
# build
make -j${PARALLEL_LEVEL} VERBOSE=1 install

################################################################################
# BUILD - Build librmm_cffi
################################################################################

logger "Build librmm_cffi..."
make rmm_python_cffi
make rmm_install_python

################################################################################
# Docs - Build RMM docs
################################################################################
make rmm_doc

cd $WORKSPACE
rm -rf ${DOCS_DIR}/*
mv doxygen/html/* $DOCS_DIR
