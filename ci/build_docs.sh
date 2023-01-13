#!/bin/bash

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
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  rmm librmm

git clone https://github.com/rapidsai/rmm.git

# Build CPP docs
gpuci_logger "Build Doxygen docs"
cd ./doxygen
doxygen Doxyfile

# Build Python docs
gpuci_logger "Build Python docs"
cd ../python/docs
sphinx-build -b html . _html
sphinx-build -b text . _text

if [[ ${RAPIDS_BUILD_TYPE} == "branch" ]]; then
  aws s3 sync _html s3://rapidsai-docs/rmm/html
  aws s3 sync _txt s3://rapidsai-docs/rmm/txt
fi
