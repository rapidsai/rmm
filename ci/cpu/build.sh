#!/usr/bin/env bash
# Copyright (c) 2020, NVIDIA CORPORATION.
######################################
# rmm CPU build script for CI        #
######################################
set -e

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Get env"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# BUILD - Conda package builds (conda deps: librmm <- rmm)
################################################################################

if [[ "$BUILD_LIBRMM" == "1" ]]; then
  gpuci_logger "Build conda pkg for librmm"
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_conda_retry build conda/recipes/librmm --python=$PYTHON
  else
    gpuci_conda_retry build --dirty --no-remove-work-dir conda/recipes/librmm
  fi
fi

if [[ "$BUILD_RMM" == "1" ]]; then
  gpuci_logger "Build conda pkg for rmm"
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_conda_retry build conda/recipes/rmm --python=$PYTHON
  else
    gpuci_conda_retry build --dirty --no-remove-work-dir conda/recipes/rmm
    conda build --dirty --no-remove-work-dir \
      -c $WORKSPACE/ci/artifacts/rmm/cpu/conda-bld/ conda/recipes/rmm

  fi
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Upload conda packages"
source ci/cpu/upload.sh

