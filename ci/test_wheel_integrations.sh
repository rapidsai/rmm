#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBRMM_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
RMM_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" rmm --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
PIP_INSTALL_SHARED_ARGS=(
    --prefer-binary
    --constraint="${PIP_CONSTRAINT}"
    "$(echo "${LIBRMM_WHEELHOUSE}"/librmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"
    "$(echo "${RMM_WHEELHOUSE}"/rmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"
)

EXITCODE=0

rapids-logger "Check GPU usage"
nvidia-smi

# Check CUDA version for PyTorch compatibility (requires CUDA 12.8+)
CUDA_MAJOR=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d'.' -f1)
CUDA_MINOR=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d'.' -f2)

echo "::group::PyTorch Tests"

# Update this when 'torch' publishes CUDA wheels supporting newer CTKs.
#
# See notes in 'dependencies.yaml' for details on supported versions.
if \
    { [ "${CUDA_MAJOR}" -eq 12 ] && [ "${CUDA_MINOR}" -ge 6 ]; } \
    || { [ "${CUDA_MAJOR}" -eq 13 ] && [ "${CUDA_MINOR}" -le 0 ]; }; \
then

    # ensure a CUDA variant of 'torch' is used
    rapids-logger "Downloading PyTorch CUDA wheels"
    TORCH_WHEEL_DIR="$(mktemp -d)"
    ./ci/download-torch-wheels.sh "${TORCH_WHEEL_DIR}"

    rapids-logger "Installing PyTorch test requirements"
    rapids-pip-retry install \
        -v \
        "${PIP_INSTALL_SHARED_ARGS[@]}" \
        "${TORCH_WHEEL_DIR}"/torch-*.whl

    timeout 15m python -m pytest -k "torch" ./python/rmm/rmm/tests \
        && EXITCODE_PYTORCH=$? || EXITCODE_PYTORCH=$?

    if [ "${EXITCODE_PYTORCH}" != "0" ]; then
        EXITCODE="${EXITCODE_PYTORCH}"
    fi
else
    rapids-logger "Skipping PyTorch tests (requires CUDA 12.6-12.9 or 13.0, found ${RAPIDS_CUDA_VERSION})"
fi

echo "::endgroup::"

echo "::group::CuPy Tests"

rapids-logger "Generating CuPy test requirements"
rapids-dependency-file-generator \
    --output requirements \
    --file-key test_wheels_cupy \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};use_cuda_wheels=true" \
    | tee test-cupy-requirements.txt

rapids-logger "Installing CuPy test requirements"
rapids-pip-retry install \
    -v \
    "${PIP_INSTALL_SHARED_ARGS[@]}" \
    -r test-cupy-requirements.txt

timeout 15m python -m pytest -k "cupy" ./python/rmm/rmm/tests \
    && EXITCODE_CUPY=$? || EXITCODE_CUPY=$?

if [ "${EXITCODE_CUPY}" != "0" ]; then
    EXITCODE="${EXITCODE_CUPY}"
fi

echo "::endgroup::"

exit "${EXITCODE}"
