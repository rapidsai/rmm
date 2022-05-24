#!/bin/bash
set -e

# Check environment
source ci/check_environment.sh

# Update env vars
source rapids-env-update

################################################################################
# BUILD - Conda package builds (RMM)
################################################################################
gpuci_logger "Begin py build"

# Python Build Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

conda build \
  -c "${CPP_CHANNEL}" \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/rmm

rapids-upload-conda-to-s3 python
