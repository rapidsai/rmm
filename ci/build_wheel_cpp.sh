#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

# TODO Erase this once rapids-telemetry-record is merged
wget https://github.com/bdice/gha-tools/archive/refs/heads/rapids-telemetry-record.tar.gz -O - | tar -xz -C /usr/local/bin --strip-components=2 gha-tools-rapids-telemetry-record/tools/*

package_dir="python/librmm"

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

cd "${package_dir}"

sccache --zero-stats

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

rapids-telemetry-record build.log rapids-pip-retry wheel . -w dist -v --no-deps --disable-pip-version-check

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats

python -m wheel tags --platform any dist/* --remove

../../ci/validate_wheel.sh dist

RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp dist
