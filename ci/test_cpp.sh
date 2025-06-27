#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# Run librmm gtests from librmm-tests package
rapids-logger "Run gtests"

export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/
./ci/run_ctests.sh -j20 && EXITCODE=$? || EXITCODE=$?;

# Run all examples from librmm-example package
for example in "${CONDA_PREFIX}"/bin/examples/librmm/*; do
    if [ -x "$example" ]; then
        rapids-logger "Running example: $(basename "$example")"
        "$example" && EXAMPLE_EXITCODE=$? || EXAMPLE_EXITCODE=$?;
        if [ "$EXAMPLE_EXITCODE" -ne 0 ]; then
            rapids-logger "Example $(basename "$example") failed with exit code: $EXAMPLE_EXITCODE"
            EXITCODE=$EXAMPLE_EXITCODE
            break
        fi
    fi
done

rapids-logger "Test script exiting with value: $EXITCODE"
exit "${EXITCODE}"
