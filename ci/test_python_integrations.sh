#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_python_integrations.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

rapids-logger "Create test conda environment"

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-package-name "conda_python" rmm --stable --cuda "$RAPIDS_CUDA_VERSION")")

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0

# Check CUDA version for PyTorch compatibility (requires CUDA 12.9+)
CUDA_MAJOR=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d'.' -f1)
CUDA_MINOR=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d'.' -f2)

echo "::group::PyTorch Tests"

if [ "${CUDA_MAJOR}" -gt 12 ] || { [ "${CUDA_MAJOR}" -eq 12 ] && [ "${CUDA_MINOR}" -ge 9 ]; }; then
    rapids-logger "pytest rmm with PyTorch"

    rapids-logger "Generate PyTorch testing dependencies"
    rapids-dependency-file-generator \
      --output conda \
      --file-key test_pytorch \
      --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
      --prepend-channel "${CPP_CHANNEL}" \
      --prepend-channel "${PYTHON_CHANNEL}" \
      | tee env.yaml

    rapids-mamba-retry env create --yes -f env.yaml -n test_pytorch

    set +u
    conda activate test_pytorch
    set -u

    rapids-print-env

    timeout 10m ./ci/run_pytests.sh \
        -k "torch" \
        --junitxml="${RAPIDS_TESTS_DIR}/junit-rmm-pytorch.xml" \
        --cov-config=.coveragerc \
        --cov=rmm \
        --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/rmm-pytorch-coverage.xml" \
        --cov-report term \
     && EXITCODE_PYTORCH=$? || EXITCODE_PYTORCH=$?;

    if [ "${EXITCODE_PYTORCH}" != "0" ]; then
        EXITCODE="${EXITCODE_PYTORCH}"
    fi
else
    rapids-logger "Skipping PyTorch tests (requires CUDA 12.9+, found ${RAPIDS_CUDA_VERSION})"
fi

echo "::endgroup::"

echo "::group::CuPy Tests"

rapids-logger "pytest rmm with CuPy"

rapids-logger "Generate CuPy testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cupy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test_cupy

set +u
conda activate test_cupy
set -u

rapids-print-env

timeout 10m ./ci/run_pytests.sh \
    -k "cupy" \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-rmm-cupy.xml" \
    --cov-config=.coveragerc \
    --cov=rmm \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/rmm-cupy-coverage.xml" \
    --cov-report term \
 && EXITCODE_CUPY=$? || EXITCODE_CUPY=$?;

if [ "${EXITCODE_CUPY}" != "0" ]; then
    EXITCODE="${EXITCODE_CUPY}"
fi

echo "::endgroup::"

rapids-logger "Test script exiting with value: $EXITCODE"
exit "${EXITCODE}"
