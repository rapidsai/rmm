#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

# TODO: remove before merging (when new rapids-build-backend is released)
if [[ ! -d /tmp/delete-me/rapids-build-backend ]]; then
    git clone \
        -b main \
        https://github.com/rapidsai/rapids-build-backend.git \
        /tmp/delete-me/rapids-build-backend

    pushd /tmp/delete-me/rapids-build-backend
    sed -e 's/^version =.*/version = "0.0.2"/' -i ./python/rmm/pyproject.toml
    python -m pip install .
    popd
fi
export PIP_FIND_LINKS="file:///tmp/delete-me/rapids-build-backend"

package_name="rmm"
package_dir="python/rmm"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
echo "${version}" > VERSION

if rapids-is-release-build; then
    export RAPIDS_ALLOW_NIGHTLY_DEPS=0
fi

# Need to manually patch the cuda-python version for CUDA 12.
ctk_major=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d'.' -f1)
if [[ ${ctk_major} == "12" ]]; then
    sed -i "s/cuda-python[<=>\.,0-9a]*/cuda-python>=12.0,<13.0a0/g" ${package_dir}/pyproject.toml
fi

cd "${package_dir}"

pip install 'scikit-build-core[pyproject]>=0.7.0'
python -m pip wheel . -w dist -vvv --no-deps --no-build-isolation --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
