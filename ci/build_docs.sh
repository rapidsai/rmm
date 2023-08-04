#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  rmm librmm

export RAPIDS_VERSION_NUMBER="23.10"
export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build CPP docs"
pushd doxygen
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/librmm/html"
mv html/* "${RAPIDS_DOCS_DIR}/librmm/html"
popd

rapids-logger "Build Python docs"
pushd python/docs
make dirhtml
make text
mkdir -p "${RAPIDS_DOCS_DIR}/rmm/"{html,txt}
mv _build/dirhtml/* "${RAPIDS_DOCS_DIR}/rmm/html"
mv _build/text/* "${RAPIDS_DOCS_DIR}/rmm/txt"
popd

rapids-upload-docs
