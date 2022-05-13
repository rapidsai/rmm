#!/bin/bash
set -e

# Check environment
source ci/check_environment.sh

################################################################################
# BUILD - Conda package builds (LIBRMM)
################################################################################
gpuci_logger "Begin cpp build"

conda build \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/librmm

rapids-upload-conda-to-s3 cpp
