#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

package_name="rmm"
package_dir="python"

if [[ ! -d "/tmp/gha-tools" ]]; then
  git clone https://github.com/vyasr/gha-tools.git -b feat/generate_versions /tmp/gha-tools
fi

version_override=$(/tmp/gha-tools/tools/rapids-generate-version)
commit_override=$(git rev-parse HEAD)

sed -i "s/__version__ = .*/__version__ = ${version_override}/g" ${package_dir}/${package_name}/__init__.py
sed -i "s/__git_commit__ = .*/__commit__ = \"${commit_override}\"/g" ${package_dir}/${package_name}/__init__.py

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# This calls mambabuild when boa is installed (as is the case in the CI images)
RAPIDS_PACKAGE_VERSION=${version_override} rapids-conda-retry mambabuild -c "${CPP_CHANNEL}" conda/recipes/rmm

rapids-upload-conda-to-s3 python
