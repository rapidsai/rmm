#!/usr/bin/env bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# rmm GPU build & testscript for CI  #
######################################
set -e

source "$WORKSPACE/ci/cpu/build.sh"

################################################################################
# Test - librmm
################################################################################

logger "Check GPU usage..."
nvidia-smi

logger "Running googletests..."

GTEST_OUTPUT="xml:${WORKSPACE}/test-results/" make -j${PARALLEL_LEVEL} test

logger "Python py.test for librmm_cffi..."
cd $WORKSPACE/python
py.test --cache-clear --junitxml=${WORKSPACE}/test-results/junit-librmm_cffi.xml -v