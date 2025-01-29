#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# rattler-build doesn't have built-in support for GIT_DESCRIBE_NUMBER and
# GIT_DESCRIBE_HASH so we set them in the environment first
# shellcheck disable=SC2034
IFS=- read -r TAG_VERSION GIT_DESCRIBE_NUMBER GIT_DESCRIBE_HASH <<< "$(git describe --tags)"
unset TAG_VERSION
export GIT_DESCRIBE_NUMBER
export GIT_DESCRIBE_HASH

mamba install rattler-build -c conda-forge -y

rattler-build build --recipe conda/recipes/rmm \
                    --experimental \
                    --no-build-id \
                    --channel-priority disabled \
                    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
                    -c "${CPP_CHANNEL}"
                    # ^^^ Probably need this, but locally `rattler-build` finds the CPP builds automatically
                    #
                    # These are probably set via `rapids-configure-conda-channels`
                    # -c rapidsai \
                    # -c conda-forge \
                    # -c nvidia

sccache --show-adv-stats

rapids-upload-conda-to-s3 python
