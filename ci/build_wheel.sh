#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

LIBRMM_WHEELHOUSE="${PWD}/wheel-output/librmm"
RMM_WHEELHOUSE="${PWD}/wheel-output/rmm"
mkdir -p "${LIBRMM_WHEELHOUSE}" "${RMM_WHEELHOUSE}"

RAPIDS_WHEEL_BLD_OUTPUT_DIR="${LIBRMM_WHEELHOUSE}" ./ci/build_wheel_cpp.sh

LIBRMM_WHEELHOUSE="${LIBRMM_WHEELHOUSE}" RAPIDS_WHEEL_BLD_OUTPUT_DIR="${RMM_WHEELHOUSE}" ./ci/build_wheel_python.sh

{
  echo "librmm_artifact_name=$(rapids-artifact-name wheel_cpp librmm rmm --cuda "${RAPIDS_CUDA_VERSION}")"
  echo "librmm_output_dir=${LIBRMM_WHEELHOUSE}"
  echo "rmm_artifact_name=$(rapids-artifact-name wheel_python rmm rmm --stable --cuda "${RAPIDS_CUDA_VERSION}")"
  echo "rmm_output_dir=${RMM_WHEELHOUSE}"
} >> "${GITHUB_OUTPUT}"
