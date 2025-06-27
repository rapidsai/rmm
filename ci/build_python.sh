#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# Creates and exports $RATTLER_CHANNELS
source rapids-rattler-channel-string

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rapids-telemetry-record build.log rattler-build build \
    --recipe conda/recipes/rmm \
    --experimental \
    --no-build-id \
    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
    -c "${CPP_CHANNEL}" \
    "${RATTLER_CHANNELS[@]}"

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats

# See https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache
