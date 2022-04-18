#!/bin/bash

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
# BUILD - Conda package builds (RMM)
################################################################################
gpuci_logger "Begin build"
CPP_FILE_NAME="conda_librmm_build_${BRANCH_NAME}-arc-${ARC}.tar"
PY_FILE_NAME="conda_rmm_build_${BRANCH_NAME}-arc-${ARC}-py-${PY_VER}.tar"
CONDA_BLD_DIR=".py-conda-bld"

# Copy Cpp artifact from s3
aws s3 cp "s3://rapids-downloads/ci/${CPP_FILE_NAME}" conda_cpp.tar
mkdir cpp_channel
tar -xvf conda_cpp.tar -C cpp_channel/
ls -la cpp_channel
find .

# Build
conda build --channel ./cpp_channel/.conda-bld conda/recipes/rmm --croot ${CONDA_BLD_DIR}

# Copy artifact to s3
tar -cvf ${PY_FILE_NAME} ${CONDA_BLD_DIR}
aws s3 cp ${PY_FILE_NAME} "s3://rapids-downloads/ci/"
