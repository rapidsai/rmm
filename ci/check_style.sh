#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create checks conda environment"

. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n checks
conda activate checks

RAPIDS_VERSION_NUMBER="$(rapids-version-major-minor)"
FORMAT_FILE_URL="https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION_NUMBER}/cmake-format-rapids-cmake.json"
RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json
export RAPIDS_CMAKE_FORMAT_FILE
mkdir -p "$(dirname ${RAPIDS_CMAKE_FORMAT_FILE})"
wget -O ${RAPIDS_CMAKE_FORMAT_FILE} "${FORMAT_FILE_URL}"

# Run pre-commit checks
pre-commit run --all-files --show-diff-on-failure
