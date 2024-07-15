#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
WHEELHOUSE="${PWD}/dist/"
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python "${WHEELHOUSE}"
PIP_PACKAGE=$(echo "${WHEELHOUSE}"/rmm_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl | head -n1)

# Use `package[test]` to install latest test dependencies or explicitly install oldest.
if [[ $RAPIDS_DEPENDENCIES != "earliest" ]]; then
  python -m pip install -v "${PIP_PACKAGE}[test]"
else
  rapids-dependency-file-generator \
      --output requirements \
      --file-key test_python \
      --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
    | tee oldest-dependencies.txt

  python -m pip install -v "$PIP_PACKAGE" -r oldest-dependencies.txt
fi

python -m pytest ./python/rmm/rmm/tests
