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

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

source rapids-configure-rattler

rattler-build build --recipe conda/recipes/rmm \
                    --experimental \
                    --no-build-id \
                    --channel-priority disabled \
                    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
                    -c "${CPP_CHANNEL}" \
                    -c rapidsai \
                    -c rapidsai-nightly \
                    -c conda-forge \
                    -c nvidia

sccache --show-adv-stats

rapids-upload-conda-to-s3 python
