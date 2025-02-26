#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

# TODO Erase this once rapids-telemetry-record is merged
wget https://github.com/bdice/gha-tools/archive/refs/heads/rapids-telemetry-record.tar.gz -O - | tar -xz - gha-tools-rapids-telemetry-record/tools --strip-components=2 -C /usr/local/bin

package_name="rmm"
package_dir="python/rmm"

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
python -m auditwheel repair "${EXCLUDE_ARGS[@]}" -w final_dist dist/*

../../ci/validate_wheel.sh final_dist

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python final_dist

# switch back to the root of the repo and check symbol visibility
popd
ci/check_symbols.sh "$(echo ${package_dir}/final_dist/rmm_*.whl)"
