#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eoxu pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# TODO: Remove when this is set in the container
PIP_EXTRA_INDEX_URL="https://pypi.k8s.rapids.ai/simple"

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install -v $(echo ./dist/rmm*.whl)[test]

# Run smoke tests for aarch64 pull requests
if [ "${arch}" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]; then
    python ./ci/wheel_smoke_test.py
else
    python -m pytest ./python/rmm/tests
fi
