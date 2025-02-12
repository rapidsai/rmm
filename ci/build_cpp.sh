#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats

# This calls mambabuild when boa is installed (as is the case in the CI images)
RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry mambabuild conda/recipes/librmm 2>&1 | tee telemetry-artifacts/build.log

sccache --show-adv-stats | tee telemetry-artifacts/sccache-stats.txt

rapids-upload-conda-to-s3 cpp
