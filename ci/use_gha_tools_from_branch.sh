#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

git clone \
  --branch "gh-artifacts/stricter-run-selection" \
  https://github.com/rapidsai/gha-tools.git \
  /tmp/gha-tools

export PATH="/tmp/gha-tools/tools":$PATH
