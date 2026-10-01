#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/build_wheel_common.sh
source ./ci/build_wheel_common.sh

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
AUDITWHEEL_EXCLUDES=(
  --exclude librapids_logger.so
  --exclude librmm.so
)

repair_wheel() {
  python -m auditwheel repair \
    "${AUDITWHEEL_EXCLUDES[@]}" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    "$@"
}

add_wheel_constraint() {
  local package_name=$1
  local wheel_glob=$2
  local -a wheel_paths=()

  # auditwheel determines the final platform/ABI tag, so resolve its output
  # filename before recording the local direct-reference constraint.
  mapfile -t wheel_paths < <(compgen -G "${wheel_glob}")
  if (( ${#wheel_paths[@]} != 1 )); then
    echo "Expected exactly one wheel matching ${wheel_glob}, found ${#wheel_paths[@]}" >&2
    exit 1
  fi

  echo "${package_name}-${RAPIDS_PY_CUDA_SUFFIX} @ file://${wheel_paths[0]}" >> "${PIP_CONSTRAINT}"
}

# librmm
build_package_wheel librmm librmm python/librmm

repair_wheel python/librmm/dist/*

finalize_package_wheel \
  librmm \
  python/librmm \
  "$(rapids-artifact-name wheel_cpp librmm rmm --cuda "${RAPIDS_CUDA_VERSION}")"

# rmm uses the librmm wheel built above.
add_wheel_constraint librmm "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/librmm_*.whl"

export RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"

# rmm
build_package_wheel rmm rmm python/rmm --stable

repair_wheel python/rmm/dist/*

./ci/check_symbols.sh "$(echo "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/rmm_*.whl)"

finalize_package_wheel \
  rmm \
  python/rmm \
  "$(rapids-artifact-name wheel_python rmm rmm --stable --cuda "${RAPIDS_CUDA_VERSION}")"
