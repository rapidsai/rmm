#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-print-env

echo "file contents" > "${RAPIDS_ARTIFACTS_DIR}"/somefile.txt
