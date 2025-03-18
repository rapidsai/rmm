#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Path to the RMM include directory (absolute path)
RMM_INCLUDE_DIR="${SCRIPT_DIR}/../../include"

echo "Testing compilation failure when LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE is not defined"
echo "Using RMM include directory: ${RMM_INCLUDE_DIR}"

# Create a temporary file for compilation errors
ERROR_FILE=$(mktemp)
trap 'rm -f "${ERROR_FILE}"' EXIT

# Try to compile the file without defining LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
set +e
g++ -std=c++17 -I"${RMM_INCLUDE_DIR}" libcudacxx_flag_test.cpp -o libcudacxx_flag_test 2> "${ERROR_FILE}"
set -e

if $?; then
  echo "Test failed: Compilation succeeded when it should have failed"
  exit 1
fi

# Check if the error message contains the expected text
if ! grep -q "RMM requires LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE to be defined" "${ERROR_FILE}"; then
  echo "Test failed: Compilation failed but with an unexpected error message:"
  cat "${ERROR_FILE}"
  exit 1
fi

# Don't show the error message, to avoid confusing it with a real error in the CI logs.
echo "Test passed: Compilation failed with the expected error message"
exit 0
