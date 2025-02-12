#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

sccache --zero-stats

# This calls mambabuild when boa is installed (as is the case in the CI images)
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild -c "${CPP_CHANNEL}" conda/recipes/rmm 2>&1 | tee telemetry-artifacts/build.log

sccache --show-adv-stats | tee telemetry-artifacts/sccache-stats.txt

rapids-upload-conda-to-s3 python
