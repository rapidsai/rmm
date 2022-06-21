#!/bin/bash

set -euo pipefail

# Update env vars
source rapids-env-update

# Check environment
rapids-check-env

rapids-logger "Begin py build"

# Python Build Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry mambabuild -c "${CPP_CHANNEL}" conda/recipes/rmm

rapids-upload-conda-to-s3 python
