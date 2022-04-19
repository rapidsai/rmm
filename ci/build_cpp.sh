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
FILE_NAME="conda_librmm_build_${BRANCH_NAME}-arc-${ARC}.tar"

# Build
conda build \
  --croot ${CONDA_BLD_DIR} \
  --no-build-id \
  conda/recipes/librmm

# Copy artifact to s3
tar -cvf ${FILE_NAME} ${CONDA_BLD_DIR}
aws s3 cp ${FILE_NAME} "s3://rapids-downloads/ci/"
