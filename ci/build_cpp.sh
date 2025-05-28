#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats

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

# remove build_cache directory
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

# Run the libcudacxx flag test at build time, since compilers are available
rapids-logger "Run libcudacxx_flag_test"
./cpp/tests/libcudacxx_flag_test/libcudacxx_flag_test.sh
