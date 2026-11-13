#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

LIBRMM_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp librmm rmm --cuda "$RAPIDS_CUDA_VERSION")")
RMM_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python rmm rmm --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

python -m venv librmm-env
. librmm-env/bin/activate

rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${LIBRMM_WHEELHOUSE}"/librmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"
python -c "import librmm; assert librmm.load_library() is not None"
deactivate

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${LIBRMM_WHEELHOUSE}"/librmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${RMM_WHEELHOUSE}"/rmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

timeout 15m python -m pytest ./python/rmm/rmm/tests
