#!/bin/bash
# Copyright (c) 2020-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin cpp build"

# This calls mambabuild when boa is installed (as is the case in the CI images)
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild conda/recipes/librmm

rapids-upload-conda-to-s3 cpp
