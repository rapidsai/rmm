#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_dir="python/rmm"

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="rmm-${RAPIDS_CONDA_ARCH}-cuda${RAPIDS_CUDA_VERSION%%.*}-wheel-preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

rapids-generate-version > ./VERSION

pushd "${package_dir}"

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")
LIBRMM_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# ensure 'rmm' wheel builds always use the 'librmm' just built in the same CI run
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRMM_WHEELHOUSE}"/librmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" >> "${PIP_CONSTRAINT}"

sccache --stop-server 2>/dev/null || true

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

RAPIDS_PIP_WHEEL_ARGS=(
  -w dist
  -v
  --no-deps
  --disable-pip-version-check
)

# Only use --build-constraint when build isolation is enabled.
#
# Passing '--build-constraint' and '--no-build-isolation` together results in an error from 'pip',
# but we want to keep environment variable PIP_CONSTRAINT set unconditionally.
# PIP_NO_BUILD_ISOLATION=0 means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/573
if [[ "${PIP_NO_BUILD_ISOLATION:-}" != "0" ]]; then
    RAPIDS_PIP_WHEEL_ARGS+=(--build-constraint="${PIP_CONSTRAINT}")
fi

rapids-telemetry-record build.log rapids-pip-retry wheel \
  "${RAPIDS_PIP_WHEEL_ARGS[@]}" \
  .

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

EXCLUDE_ARGS=(
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
)
python -m auditwheel repair \
    "${EXCLUDE_ARGS[@]}" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    dist/*

../../ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

absolute_wheel_dir=$(realpath "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}")
# switch back to the root of the repo and check symbol visibility
popd
ci/check_symbols.sh "$(echo "${absolute_wheel_dir}"/rmm_*.whl)"
