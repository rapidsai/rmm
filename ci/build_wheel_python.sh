#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="rmm"
package_dir="python/rmm"

wheel_dir=${RAPIDS_WHEEL_DIR:-"final_dist"}

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

pushd "${package_dir}"

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# ensure 'rmm' wheel builds always use the 'librmm' just built in the same CI run
#
# using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when created the isolated build environment
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${CPP_WHEELHOUSE}"/librmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" > ./build-constraints.txt

sccache --zero-stats

PIP_CONSTRAINT="${PWD}/build-constraints.txt" \
    python -m pip wheel . -w dist -v --no-deps --disable-pip-version-check

sccache --show-adv-stats

mkdir -p final_dist
python -m auditwheel repair -w "${wheel_dir}" dist/*

../../ci/validate_wheel.sh "${wheel_dir}"

# RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python final_dist

absolute_wheel_dir=$(realpath "${wheel_dir}")
# switch back to the root of the repo and check symbol visibility
popd
ci/check_symbols.sh "$(echo ${absolute_wheel_dir}/rmm_*.whl)"
