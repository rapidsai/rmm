#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild conda/recipes/librmm

rapids-upload-conda-to-s3 cpp
