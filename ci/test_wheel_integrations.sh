#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX="true"
export RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBRMM_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
RMM_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python ./constraints.txt

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
#     ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --constraint ./constraints.txt \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${LIBRMM_WHEELHOUSE}"/librmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${RMM_WHEELHOUSE}"/rmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

EXITCODE=0

echo "::group::PyTorch Tests"

rapids-logger "Generating PyTorch test requirements"
rapids-dependency-file-generator \
    --output requirements \
    --file-key test_pytorch \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
    | tee test-pytorch-requirements.txt

rapids-logger "Installing PyTorch test requirements"
rapids-pip-retry install \
    -v \
    -r test-pytorch-requirements.txt

timeout 15m python -m pytest -k "torch" ./python/rmm/rmm/tests \
    && EXITCODE_PYTORCH=$? || EXITCODE_PYTORCH=$?

if [ "${EXITCODE_PYTORCH}" != "0" ]; then
    EXITCODE="${EXITCODE_PYTORCH}"
fi

echo "::endgroup::"

echo "::group::CuPy Tests"

rapids-logger "Generating CuPy test requirements"
rapids-dependency-file-generator \
    --output requirements \
    --file-key test_cupy \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
    | tee test-cupy-requirements.txt

rapids-logger "Installing CuPy test requirements"
rapids-pip-retry install \
    -v \
    -r test-cupy-requirements.txt

timeout 15m python -m pytest -k "cupy" ./python/rmm/rmm/tests \
    && EXITCODE_CUPY=$? || EXITCODE_CUPY=$?

if [ "${EXITCODE_CUPY}" != "0" ]; then
    EXITCODE="${EXITCODE_CUPY}"
fi

echo "::endgroup::"

exit "${EXITCODE}"
