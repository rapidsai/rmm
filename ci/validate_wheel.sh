#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

wheel_dir_relative_path=$1

rapids-logger "validate packages with 'pydistcheck'"

pydistcheck \
    --inspect \
    "${wheel_dir_relative_path}"/*.whl

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "${wheel_dir_relative_path}"/*.whl

rapids-logger "validate packages with 'abi3audit'"

# 'abi3audit' fails on wheels with DSOs that lack an ABI tag (e.g. 'lib*' wheels).
# Filtering by '*abi*' avoids those.
find \
    "${wheel_dir_relative_path}" \
    -type f \
    -name '*abi*' \
    -exec abi3audit --strict --summary --verbose '{}' \+
