#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
WHEELHOUSE="${PWD}/dist/"
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp "${WHEELHOUSE}"
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python "${WHEELHOUSE}"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python ./constraints.txt

# echo to expand wildcard before adding '[extra]' requires for pip
python -m pip install \
    -v \
    --constraint ./constraints.txt \
    "$(echo "${WHEELHOUSE}"/librmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${WHEELHOUSE}"/rmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

python -m pytest ./python/rmm/rmm/tests
