#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

# Any failing command will set EXITCODE to non-zero
set -e           # abort the script on error, this will change for running tests (see below)
set -o pipefail  # piped commands propagate their error
set -E           # ERR traps are inherited by subcommands
trap "EXITCODE=1" ERR

. /opt/conda/etc/profile.d/conda.sh
conda activate base

rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  librmm librmm-tests

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"
EXITCODE=0

rapids-logger "Check GPU usage"
nvidia-smi

# Do not abort the script on error from this point on. This allows all tests to
# run regardless of pass/fail, but relies on the ERR trap above to manage the
# EXITCODE for the script.
set +e

rapids-logger "Running googletests"
for gt in "$CONDA_PREFIX/bin/gtests/librmm/"* ; do
    ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR}/
done

echo "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
