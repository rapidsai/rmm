#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/librmm/"

# Run gtest verbosely
./DEVICE_MR_REF_TEST --gtest_filter="ResourceTests/mr_ref_test.SetCurrentDeviceResourceRef/System"

#ctest --no-tests=error --output-on-failure "$@"
