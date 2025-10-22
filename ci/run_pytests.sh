#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

# It is essential to cd into python/rmm/rmm as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/rmm/rmm/

pytest --cache-clear -v "$@" tests
