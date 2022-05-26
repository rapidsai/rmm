#!/bin/bash
set -e

if [ ${ARCH} == "arm64" ]; then
  exit 1
fi

# Check environment
source ci/check_environment.sh

# Update env vars
source rapids-env-update

################################################################################
# BUILD - Conda package builds (LIBRMM)
################################################################################
gpuci_logger "Begin cpp build"

conda build \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/librmm

rapids-upload-conda-to-s3 cpp
