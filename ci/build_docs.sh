#!/bin/bash

VERSION_NUMBER=$(echo `git describe --abbrev=0 --tags` | grep -o -E '([0-9]+\.[0-9]+)')

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
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

# Build CPP docs
rapids-logger "Build Doxygen docs"
cd ./doxygen
doxygen Doxyfile

# Build Python docs
rapids-logger "Build Python docs"
cd ../python/docs
sphinx-build -b dirhtml . _html
sphinx-build -b text . _text

if [[ ${RAPIDS_BUILD_TYPE} == "branch" ]]; then
  aws s3 sync --delete _html "s3://rapidsai-docs/rmm/${VERSION_NUMBER}/html"
  aws s3 sync --delete _text "s3://rapidsai-docs/rmm/${VERSION_NUMBER}/txt"
  aws s3 sync --delete ../../doxygen/html "s3://rapidsai-docs/librmm/${VERSION_NUMBER}/html"
fi
