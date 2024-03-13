#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="rmm"
package_dir="python"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
echo "${version}" > VERSION

if rapids-is-release-build; then
    export RAPIDS_ONLY_RELEASE_DEPS=1
fi

# Need to manually patch the cuda-python version for CUDA 12.
ctk_major=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d'.' -f1)
if [[ ${ctk_major} == "12" ]]; then
    sed -i "s/cuda-python[<=>\.,0-9a]*/cuda-python>=12.0,<13.0a0/g" ${package_dir}/pyproject.toml
fi

cd "${package_dir}"

# For testing, will be removed when the package is released
git clone https://github.com/rapidsai/rapids-build-backend.git
pushd rapids-build-backend
python -m pip wheel . --no-deps
popd

PIP_FIND_LINKS="file://${PWD}/rapids-build-backend/" python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
