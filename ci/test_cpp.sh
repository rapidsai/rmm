#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  librmm librmm-tests

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run librmm gtests from librmm-tests package
rapids-logger "Run gtests"

cd $CONDA_PREFIX/bin/gtests/librmm/
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/
ctest -j20 --output-on-failure

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
