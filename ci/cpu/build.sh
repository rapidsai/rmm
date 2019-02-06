#!/usr/bin/env bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# cuDF CPU conda build script for CI #
######################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# INSTALL - Install NVIDIA driver
################################################################################

if [ "$INSTALL_DRIVER" -ne "0" ]; then
    logger "Install NVIDIA driver for CUDA $CUDA..."
    apt-get update -q
    DRIVER_VER="396.44-1"
    LIBCUDA_VER="396"
    if [ "$CUDA" == "10.0" ]; then
      DRIVER_VER="410.72-1"
      LIBCUDA_VER="410"
    fi
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cuda-drivers=${DRIVER_VER} libcuda1-${LIBCUDA_VER}
else
    logger "Skipping driver install..."
fi

################################################################################
# BUILD - Build librmm
################################################################################

logger "Build librmm..."
CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX11_ABI=$BUILD_ABI"

# Use CMake-based build procedure
mkdir -p build
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