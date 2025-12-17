#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

# Download custom rapids-package-name and add to PATH
mkdir -p /tmp/gha-tools
curl -fsSL https://raw.githubusercontent.com/gforsyth/gha-tools/e956ec25ec9cb421ad90ee7407262374491625e2/tools/rapids-package-name \
  -o /tmp/gha-tools/rapids-package-name
chmod +x /tmp/gha-tools/rapids-package-name
export PATH="/tmp/gha-tools:$PATH"

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBRMM_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
    RMM_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" rmm --stable --cuda "$RAPIDS_CUDA_VERSION")")
else
    RMM_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
fi

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

timeout 15m python -m pytest ./python/rmm/rmm/tests
