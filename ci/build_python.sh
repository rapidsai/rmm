#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

package_name="rmm"
package_dir="python/rmm"

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_dir}/${package_name}/_version.py"

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# This calls mambabuild when boa is installed (as is the case in the CI images)
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild -c "${CPP_CHANNEL}" conda/recipes/rmm

rapids-upload-conda-to-s3 python
