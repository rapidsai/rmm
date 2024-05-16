#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
WHEELHOUSE="${PWD}/dist/"
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python "${WHEELHOUSE}"

# echo to expand wildcard before adding '[extra]' requires for pip
python -m pip install "rmm-${RAPIDS_PY_CUDA_SUFFIX}[test]>=0.0.0a0" --find-links "${WHEELHOUSE}"

python -m pytest ./python/rmm/rmm/tests
