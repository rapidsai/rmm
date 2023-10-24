#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
set +u
conda activate test
set -u

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

rapids-logger "Check GPU usage"
nvidia-smi

cd python

EXITCODE=0
trap "EXITCODE=1" ERR
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

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
