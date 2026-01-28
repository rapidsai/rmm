#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

sccache --stop-server 2>/dev/null || true

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# Creates and exports $RATTLER_CHANNELS
source rapids-rattler-channel-string

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

# Only use stable ABI package naming for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  RECIPE_PATH="conda/recipes/rmm"
  # Also export name to use for GitHub artifact key
  RAPIDS_PACKAGE_NAME="$(rapids-package-name conda_python rmm --stable --cuda)"
  export RAPIDS_PACKAGE_NAME
else
  RECIPE_PATH="conda/recipes/rmm_py310"
fi

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rapids-telemetry-record build.log rattler-build build \
    --recipe "${RECIPE_PATH}" \
    --experimental \
    --no-build-id \
    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
    -c "${CPP_CHANNEL}" \
    "${RATTLER_CHANNELS[@]}"

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# See https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

# Only use stable ABI package naming for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  RAPIDS_PACKAGE_NAME="$(rapids-package-name conda_python rmm --stable --cuda)"
  export RAPIDS_PACKAGE_NAME
fi
