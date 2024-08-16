#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
WHEELHOUSE="${PWD}/dist/"
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python "${WHEELHOUSE}"

# Constraint to minimum dependency versions if job is set up as "oldest"
echo "" > ./constraints.txt
if [[ $RAPIDS_DEPENDENCIES == "oldest" ]]; then
  rapids-dependency-file-generator \
        --output requirements \
        --file-key test_python \
        --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
      | tee ./constraints.txt
fi

# echo to expand wildcard before adding '[extra]' requires for pip
python -m pip install \
    -v \
    --constraint ./constraints.txt \
    "$(echo "${WHEELHOUSE}"/rmm_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]"

python -m pytest ./python/rmm/rmm/tests
