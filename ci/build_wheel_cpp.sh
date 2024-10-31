#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/librmm"

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

cd "${package_dir}"

sccache --zero-stats

python -m pip wheel . -w dist -v --no-deps --disable-pip-version-check

sccache --show-adv-stats

python -m pip install wheel
python -m wheel tags --platform any dist/* --remove
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp dist
