#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/librmm"
wheel_dir=${RAPIDS_WHEEL_DIR:-"dist"}

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

cd "${package_dir}"

sccache --zero-stats

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

rapids-telemetry-record build.log rapids-pip-retry wheel . -w "${wheel_dir}" -v --no-deps --disable-pip-version-check

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats

python -m wheel tags --platform any "${wheel_dir}"/* --remove

../../ci/validate_wheel.sh "${wheel_dir}"
