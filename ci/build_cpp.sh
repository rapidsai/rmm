#!/bin/bash

set -euo pipefail

# Update env vars
source rapids-env-update

# Check env
rapids-check-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild conda/recipes/librmm

rapids-upload-conda-to-s3 cpp
