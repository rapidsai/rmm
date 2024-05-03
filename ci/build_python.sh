#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

# TODO: remove before merging (when new rapids-build-backend is released)
if [[ ! -d /tmp/delete-me/rapids-build-backend ]]; then
    git clone \
        -b main \
        https://github.com/rapidsai/rapids-build-backend.git \
        /tmp/delete-me/rapids-build-backend

    pushd /tmp/delete-me/rapids-build-backend
    sed -e 's/^version =.*/version = "0.0.2"/' -i pyproject.toml
    python -m pip install .
    popd
fi
export PIP_FIND_LINKS="file:///tmp/delete-me/rapids-build-backend"

rapids-print-env

version=$(rapids-generate-version)

echo "${version}" > VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# This calls mambabuild when boa is installed (as is the case in the CI images)
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild -c "${CPP_CHANNEL}" conda/recipes/rmm

rapids-upload-conda-to-s3 python
