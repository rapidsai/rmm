#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/librmm"

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

rapids-generate-version > ./VERSION

cd "${package_dir}"

sccache --zero-stats

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

dist_dir="$(mktemp -d)"
rapids-telemetry-record build.log rapids-pip-retry wheel . -w "${dist_dir}" -v --no-deps --disable-pip-version-check

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats

python -m auditwheel repair \
    --exclude librapids_logger.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    "${dist_dir}"/*

../../ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
