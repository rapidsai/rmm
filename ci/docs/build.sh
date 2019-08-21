#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# rmm GPU build & testscript for CI  #
######################################
set -ex

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export HOME=$WORKSPACE
export DOCS_DIR=/data/docs/html

while getopts "d" option; do
    case ${option} in
        d)
            DOCS_DIR=${OPTARG}
            ;;
    esac
done

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf
conda install -c conda-forge doxygen

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

################################################################################
# BUILD - Build and install librmm and rmm
################################################################################
logger "Build and install librmm and rmm..."
"$WORKSPACE/build.sh" -v clean librmm rmm

################################################################################
# Docs - Build RMM docs
################################################################################
make rmm_doc

cd $WORKSPACE
rm -rf ${DOCS_DIR}/*
mv doxygen/html/* $DOCS_DIR
