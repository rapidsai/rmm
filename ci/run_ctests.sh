#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

if [ -d "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/librmm/" ]; then
    # Support customizing the ctests' install location
    cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/librmm/"
    ctest --no-tests=error --output-on-failure "$@"
fi
