#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eoxu pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

PIP_EXTRA_INDEX_URL="https://pypi.k8s.rapids.ai/simple"
python -m pip install -v ./dist/rmm*.whl[test]
python -m pip check

if [ "${arch}" == "x86_64" ]; then
    python -m pytest ./python/rmm/tests
elif [ "${arch}" == "aarch64" ]; then
    python ./ci/wheel_smoke_test.py
fi
