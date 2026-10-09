#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [[ "${RAPIDS_WHEEL_COMMON_INITIALIZED:-}" != "true" ]]; then
  source rapids-configure-sccache
  source rapids-datetime-string
  source rapids-init-pip

  export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

  RAPIDS_VERSION_SUFFIX=".post${RAPIDS_DATETIME_STRING}" \
    rapids-generate-version > ./VERSION

  # Creates artifacts directory for telemetry
  source rapids-telemetry-setup

  export RAPIDS_WHEEL_COMMON_INITIALIZED=true
fi

build_wheel() (
  local package_name=$1
  local package_dir=$2
  shift 2

  local stable_abi=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --stable)
        stable_abi=true
        shift
        ;;
      *)
        echo "Unknown option: $1" >&2
        exit 1
        ;;
    esac
  done

  cd "${package_dir}" || exit

  sccache --stop-server 2>/dev/null || true

  rapids-logger "Building '${package_name}' wheel"

  local build_env_dir="/tmp/${package_name}-wheel-build-env"
  # `build` preserves this environment after a failed build for debugging. CI
  # retries must start with an empty directory, as required by `--env-dir`.
  rm -rf "${build_env_dir}"

  local -a rapids_build_args=(
    --wheel
    --outdir dist
    --verbose
    # A fixed location keeps isolated-build include paths stable for sccache.
    --env-dir "${build_env_dir}"
    --dependency-constraints-txt "${PIP_CONSTRAINT}"
  )

  if [[ "${stable_abi}" == "true" ]] && [[ -n "${RAPIDS_PY_API:-}" ]]; then
    rapids_build_args+=(--config-setting="skbuild.wheel.py-api=${RAPIDS_PY_API}")
  fi

  # `build` receives the same generated constraints explicitly. Unset the
  # environment variable so it does not constrain the frontend installation.
  unset PIP_CONSTRAINT

  rapids-telemetry-record "build-${package_name}.log" rapids-python-build-retry \
    "${rapids_build_args[@]}" \
    .

  rapids-telemetry-record "sccache-stats-${package_name}.txt" sccache --show-adv-stats
  sccache --stop-server >/dev/null 2>&1 || true
)

build_package_wheel() {
  local package_key=$1
  local package_name=$2
  local package_dir=$3
  shift 3

  export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="${package_name}-${RAPIDS_CONDA_ARCH}-cuda${RAPIDS_CUDA_VERSION%%.*}-wheel-preprocessor-cache"
  export RAPIDS_WHEEL_BLD_OUTPUT_DIR="${PWD}/wheel-output/${package_key}"
  mkdir -p "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  build_wheel "${package_name}" "${package_dir}" "$@"
}

record_wheel_artifact() {
  local package_key=$1
  local package_name=$2
  {
    echo "${package_key}_artifact_name=${package_name}"
    echo "${package_key}_output_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  } >> "${GITHUB_OUTPUT}"
}

finalize_package_wheel() {
  local package_key=$1
  local package_dir=$2
  local artifact_name=$3

  # pydistcheck reads its configuration from the package's pyproject.toml.
  (cd "${package_dir}" && ../../ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}")
  record_wheel_artifact "${package_key}" "${artifact_name}"
}
