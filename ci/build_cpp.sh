#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

# rattler-build doesn't have built-in support for GIT_DESCRIBE_NUMBER and
# GIT_DESCRIBE_HASH so we set them in the environment first
# shellcheck disable=SC2034
IFS=- read -r TAG_VERSION GIT_DESCRIBE_NUMBER GIT_DESCRIBE_HASH <<< "$(git describe --tags)"
unset TAG_VERSION
export GIT_DESCRIBE_NUMBER
export GIT_DESCRIBE_HASH

mamba install rattler-build -c conda-forge -y

rattler-build build --recipe conda/recipes/librmm \
                    --experimental \
                    --no-build-id \
                    --channel-priority disabled

                    # This is probably set via `CONDA_BLD_PATH`
                    # --output_dir /tmp/conda-bld-output
                    # These are probably set via `rapids-configure-conda-channels`
                    # -c rapidsai \
                    # -c conda-forge \
                    # -c nvidia

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp
