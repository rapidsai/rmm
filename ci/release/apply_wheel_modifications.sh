#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version>

VERSION=${1}
CUDA_SUFFIX=${2}

sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/rmm/__init__.py
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/setup.py

sed -i "s/name=\"rmm\",/name=\"rmm-${CUDA_SUFFIX}\"/g" python/setup.py
