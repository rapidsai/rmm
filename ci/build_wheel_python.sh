#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="rmm"
package_dir="python/rmm"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

if [[ ! -d "/tmp/gha-tools" ]]; then
    git clone https://github.com/msarahan/gha-tools.git -b get-pr-wheel-artifact /tmp/gha-tools
fi

RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" /tmp/gha-tools/tools/rapids-download-wheels-from-s3 cpp /tmp/librmm_dist

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

pyproject_file="${package_dir}/pyproject.toml"

sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_dir}/${package_name}/_version.py"

alpha_spec=''
if ! rapids-is-release-build; then
    alpha_spec=',>=0.0.0a0'
fi

sed -r -i "s/librmm==(.*)\"/librmm${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cuda-python[<=>\.,0-9a]*/cuda-python>=12.0,<13.0a0/g" ${pyproject_file}
fi

cd "${package_dir}"

PIP_FIND_LINKS="/tmp/librmm_dist" python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" /tmp/gha-tools/tools/rapids-upload-wheels-to-s3 python final_dist
