#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

if [[ ! -d "/tmp/gha-tools" ]]; then
  git clone https://github.com/vyasr/gha-tools.git -b feat/generate_versions /tmp/gha-tools
fi

version_override=$(/tmp/gha-tools/tools/rapids-generate-version)

rapids-logger "Begin cpp build"

# This calls mambabuild when boa is installed (as is the case in the CI images)
RAPIDS_PACKAGE_VERSION=${version_override} rapids-conda-retry mambabuild conda/recipes/librmm

rapids-upload-conda-to-s3 cpp
