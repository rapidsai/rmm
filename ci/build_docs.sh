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
VERSION_NUMBER="23.06"

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  rmm librmm

# Build CPP docs
rapids-logger "Build Doxygen docs"
pushd doxygen
doxygen Doxyfile
popd

# Build Python docs
rapids-logger "Build Python docs"
pushd python/docs
sphinx-build -b dirhtml . _html
sphinx-build -b text . _text
popd

if [[ "${RAPIDS_BUILD_TYPE}" == "branch" ]]; then
  rapids-logger "Upload Docs to S3"
  aws s3 sync --no-progress --delete doxygen/html "s3://rapidsai-docs/librmm/${VERSION_NUMBER}/html"
  aws s3 sync --no-progress --delete python/docs/_html "s3://rapidsai-docs/rmm/${VERSION_NUMBER}/html"
  aws s3 sync --no-progress --delete python/docs/_text "s3://rapidsai-docs/rmm/${VERSION_NUMBER}/txt"
fi
