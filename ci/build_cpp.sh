#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck disable=SC2155
export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tools:${PATH}"

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --stop-server 2>/dev/null || true

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

# Creates and exports $RATTLER_CHANNELS
source rapids-rattler-channel-string

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rapids-telemetry-record build.log rattler-build build \
    --recipe conda/recipes/librmm \
    --experimental \
    --no-build-id \
    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
    "${RATTLER_CHANNELS[@]}"

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# remove build_cache directory
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_cpp librmm rmm --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
