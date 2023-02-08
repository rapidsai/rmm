#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

# Any failing command will set EXITCODE to non-zero
set -e           # abort the script on error, this will change for running tests (see below)
set -o pipefail  # piped commands propagate their error
set -E           # ERR traps are inherited by subcommands
trap "EXITCODE=1" ERR

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  rmm librmm

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"
EXITCODE=0

rapids-logger "Check GPU usage"
nvidia-smi

cd python

# Do not abort the script on error from this point on. This allows all tests to
# run regardless of pass/fail, but relies on the ERR trap above to manage the
# EXITCODE for the script.
set +e

rapids-logger "pytest rmm"
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-rmm.xml" \
  -v \
  --cov-config=.coveragerc \
  --cov=rmm \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/rmm-coverage.xml" \
  --cov-report term

echo "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
