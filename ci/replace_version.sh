#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="${1}"
package_dir="${2}"

dunamai_format="{base}"
if ! rapids-is-release-build; then
    # Nightlies include the distance from the last tag as the alpha version.
    dunamai_format="{base}{stage}{distance}"
fi

# Now change the version.
python -m pip install dunamai  > /dev/null # TODO: Install into images
dunamai_version=$(python -m dunamai from git --format \"${dunamai_format}\")

echo "__version__ = '$(dunamai from any)'" > ${package_dir}/${package_name}/_version.py
