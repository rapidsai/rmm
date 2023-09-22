#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

package_name="rmm"
package_dir="python"

version_override=$(./ci/get_version.sh ${package_name} ${package_dir})

rapids-logger "Begin cpp build"

# This calls mambabuild when boa is installed (as is the case in the CI images)
RAPIDS_PACKAGE_VERSION=${version_override} rapids-conda-retry mambabuild conda/recipes/librmm

rapids-upload-conda-to-s3 cpp
