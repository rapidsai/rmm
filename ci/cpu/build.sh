#!/usr/bin/env bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# rmm CPU build script for CI        #
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

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

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
# BUILD - Conda package builds (conda deps: librmm <- rmm)
################################################################################
if [[ "$BUILD_LIBRMM" == "1" ]]; then
  logger "Build conda pkg for librmm..."
  source ci/cpu/librmm/build_librmm.sh
fi

if [[ "$BUILD_RMM" == "1" ]]; then
  logger "Build conda pkg for rmm..."
  source ci/cpu/rmm/build_rmm.sh
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

if [[ "$BUILD_LIBRMM" == "1" ]]; then
  logger "Upload conda pkg for librmm..."
  source ci/cpu/librmm/upload-anaconda.sh
fi

if [[ "$BUILD_RMM" == "1" ]]; then
  logger "Upload conda pkg for rmm..."
  source ci/cpu/rmm/upload-anaconda.sh
fi
