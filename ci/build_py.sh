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
FILE_NAME="conda_build_${BRANCH_NAME}-arc-${ARC}.tar"

aws s3 cp "s3://rapids-downloads/blobs/${FILE_NAME}" conda_cpp.tar
tar -xvf conda_cpp.tar -C cpp_channel/
conda build --channel ./cpp_channel/ conda/recipes/rmm