#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
version_override=$(./ci/get_version.sh ${package_name} ${package_dir})
#sed -i "s/^version = .*/version = ${version_override}/g" ${pyproject_file}
echo "__version__ = ${version_override}" > ${package_dir}/${package_name}/_version.py

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# This calls mambabuild when boa is installed (as is the case in the CI images)
RAPIDS_PACKAGE_VERSION=version_override rapids-conda-retry mambabuild -c "${CPP_CHANNEL}" conda/recipes/rmm

rapids-upload-conda-to-s3 python
