#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

RAPIDS_PIP_WHEEL_ARGS=(
  -w "${dist_dir}"
  -v
  --no-deps
  --disable-pip-version-check
  --build-constraint="${PIP_CONSTRAINT}"
)

# unset PIP_CONSTRAINT (set by rapids-init-pip)... it doesn't affect builds as of pip 25.3, and
# results in an error from 'pip wheel' when set and --build-constraint is also passed
unset PIP_CONSTRAINT
rapids-telemetry-record build.log rapids-pip-retry wheel \
  "${RAPIDS_PIP_WHEEL_ARGS[@]}" \
  .

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

python -m auditwheel repair \
    --exclude librapids_logger.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    "${dist_dir}"/*

../../ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
