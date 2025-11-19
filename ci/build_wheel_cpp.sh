#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_dir="python/librmm"

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="librmm-${RAPIDS_CONDA_ARCH}-cuda${RAPIDS_CUDA_VERSION%%.*}-wheel-preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

rapids-generate-version > ./VERSION

cd "${package_dir}"

sccache --stop-server 2>/dev/null || true

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

dist_dir="$(mktemp -d)"
rapids-telemetry-record build.log rapids-pip-retry wheel . -w "${dist_dir}" -v --no-deps --disable-pip-version-check

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

python -m auditwheel repair \
    --exclude librapids_logger.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    "${dist_dir}"/*

../../ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
