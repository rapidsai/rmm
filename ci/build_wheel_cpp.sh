#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="librmm"
package_dir="python/librmm"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="${package_dir}/pyproject.toml"

sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_dir}/${package_name}/_version.py"

cd "${package_dir}"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

if [[ ! -d "/tmp/gha-tools" ]]; then
    git clone https://github.com/msarahan/gha-tools.git -b get-pr-wheel-artifact /tmp/gha-tools
fi
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" /tmp/gha-tools/tools/rapids-upload-wheels-to-s3 cpp dist
