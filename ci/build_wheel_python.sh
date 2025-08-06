#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/rmm"

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

rapids-generate-version > ./VERSION

pushd "${package_dir}"

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")
LIBRMM_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# ensure 'rmm' wheel builds always use the 'librmm' just built in the same CI run
#
# Using env variable PIP_CONSTRAINT (initialized by 'rapids-init-pip') is necessary to ensure the constraints
# are used when creating the isolated build environment.
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRMM_WHEELHOUSE}"/librmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" >> "${PIP_CONSTRAINT}"

sccache --zero-stats

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

rapids-telemetry-record build.log rapids-pip-retry wheel \
  -v \
  -w dist \
  --no-deps \
  --disable-pip-version-check \
  .

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats

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
