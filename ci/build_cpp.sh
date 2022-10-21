#!/bin/bash

set -euo pipefail

source rapids-env-update

# cat <<< '
# auto_update_conda: False
# ssl_verify: False
# channels:
#   - gpuci
#   - rapidsai-nightly
#   - dask/label/dev
#   - rapidsai
#   - pytorch
#   - conda-forge
#   - nvidia
# conda-build:
#   set_build_id: false
#   root_dir: /tmp/conda-bld-workspace
#   output_folder: /tmp/conda-bld-output
# ' > /opt/conda/.condarc

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild conda/recipes/librmm

rapids-upload-conda-to-s3 cpp
