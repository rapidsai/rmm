#!/bin/bash
set -e

# Update env vars
source rapids-env-update

# Check environment
source ci/check_environment.sh

################################################################################
# BUILD - Conda package builds (LIBRMM)
################################################################################
gpuci_logger "Begin cpp build"

gpuci_conda_retry mambabuild \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/librmm

rapids-upload-conda-to-s3 cpp
