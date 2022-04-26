#!/bin/bash
set -e

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

################################################################################
# BUILD - Conda package builds (LIBRMM)
################################################################################

CONDA_BLD_DIR=".conda-bld"
LIB_TAR="librmm_${ARCH}.tar"

# FIX ME: This path is to be dynamically computed based on env vars and vary per build type.
# We will have a utility tool that consolidates the logic to compute the correct path.
FILE_PATH="ci/rmm/pull-request/${CHANGE_ID}/${GIT_COMMIT}/${LIB_TAR}"

# Build
conda build \
  --croot ${CONDA_BLD_DIR} \
  --no-build-id \
  conda/recipes/librmm

# Copy artifact to s3
tar -cvf ${LIB_TAR} ${CONDA_BLD_DIR}
aws s3 cp ${LIB_TAR} "s3://rapids-downloads/${FILE_PATH}"
