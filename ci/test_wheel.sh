#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

# TODO(jameslamb): revert before merging
git clone --branch generate-pip-constraints \
    https://github.com/rapidsai/gha-tools.git \
    /tmp/gha-tools

export PATH="/tmp/gha-tools/tools:${PATH}"

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
rapids-pip-retry install \
    -v \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${LIBRMM_WHEELHOUSE}"/librmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${RMM_WHEELHOUSE}"/rmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

timeout 15m python -m pytest ./python/rmm/rmm/tests
