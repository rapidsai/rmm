#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

env | sort

RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"

# Test latest script changes
wget \
  https://raw.githubusercontent.com/ajschmidt8/gha-tools/upload-artifacts-dir/tools/rapids-upload-artifacts-dir \
  -O /usr/local/bin/rapids-upload-artifacts-dir

echo "file contents" > "${RAPIDS_ARTIFACTS_DIR}"/somefile.txt
echo "more file contents" > "${RAPIDS_ARTIFACTS_DIR}"/another_file.txt
