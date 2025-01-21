#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"

. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"
RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION_MAJOR_MINOR

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  "rmm=${RAPIDS_VERSION}" \
  "librmm=${RAPIDS_VERSION}"

RAPIDS_DOCS_DIR="$(mktemp -d)"
export RAPIDS_DOCS_DIR


rapids-logger "Build CPP docs"
pushd doxygen
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/librmm/html"
mv html/* "${RAPIDS_DOCS_DIR}/librmm/html"
popd

rapids-logger "Build Python docs"
pushd python/rmm/docs
make dirhtml
mkdir -p "${RAPIDS_DOCS_DIR}/rmm/html"
mv _build/dirhtml/* "${RAPIDS_DOCS_DIR}/rmm/html"
popd

RAPIDS_VERSION_NUMBER="${RAPIDS_VERSION_MAJOR_MINOR}" rapids-upload-docs
