#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version> <cuda_suffix>

VERSION=${1}
CUDA_SUFFIX=${2}

sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/pyproject.toml

sed -i "s/^name = \"rmm\"/name = \"rmm${CUDA_SUFFIX}\"/g" python/pyproject.toml

if [[ $CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "/cuda-python/ s/>=11.7.1,<12.0/>=12.1,<13.0/g" python/pyproject.toml
fi
