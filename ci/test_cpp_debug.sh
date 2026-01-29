#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_cpp_debug.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

source rapids-configure-sccache

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "Generate C++ build and test dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key all \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Building librmm in Debug mode"
cmake -S cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON \
  -GNinja

cmake --build cpp/build -j

rapids-logger "Run gtests"
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/
cd cpp/build
ctest --no-tests=error --output-on-failure -j20 && EXITCODE=$? || EXITCODE=$?

sccache --show-adv-stats

rapids-logger "Test script exiting with value: $EXITCODE"
exit "${EXITCODE}"
