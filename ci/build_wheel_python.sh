#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="rmm"
package_dir="python/rmm"

wheel_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR:-"final_dist"}

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

pushd "${package_dir}"

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/librmm_dist)

# ensure 'rmm' wheel builds always use the 'librmm' just built in the same CI run
#
# using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when created the isolated build environment
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${CPP_WHEELHOUSE}"/librmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" > ./build-constraints.txt

sccache --zero-stats

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

PIP_CONSTRAINT="${PWD}/build-constraints.txt" \
    rapids-telemetry-record build.log rapids-pip-retry wheel . -w dist -v --no-deps --disable-pip-version-check

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats

mkdir -p final_dist
EXCLUDE_ARGS=(
  --exclude "librapids_logger.so"
)
python -m auditwheel repair "${EXCLUDE_ARGS[@]}" -w "${wheel_dir}" dist/*

../../ci/validate_wheel.sh "${wheel_dir}"

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python "${wheel_dir}"

absolute_wheel_dir=$(realpath "${wheel_dir}")
# switch back to the root of the repo and check symbol visibility
popd
ci/check_symbols.sh "$(echo ${absolute_wheel_dir}/rmm_*.whl)"
