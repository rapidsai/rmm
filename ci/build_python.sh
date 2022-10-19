#!/bin/bash

set -euo pipefail

source rapids-env-update

cat <<< '
auto_update_conda: False
ssl_verify: False
channels:
  - gpuci
  - rapidsai-nightly
  - dask/label/dev
  - rapidsai
  - pytorch
  - conda-forge
  - nvidia
conda-build:
  set_build_id: false
' > /opt/conda/.condarc

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry mambabuild -c "${CPP_CHANNEL}" conda/recipes/rmm

rapids-upload-conda-to-s3 python
