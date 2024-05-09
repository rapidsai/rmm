#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/librmm"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

echo "${version}" > VERSION

cd "${package_dir}"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check
python -m pip install wheel
python -m wheel tags --platform any dist/* --remove
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp dist
