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

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

source rapids-configure-rattler

rattler-build build --recipe conda/recipes/librmm \
                    --experimental \
                    --no-build-id \
                    --channel-priority disabled \
                    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR"

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp
