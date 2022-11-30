#!/bin/bash

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh
conda activate base

rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*}" > env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  librmm librmm-tests

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"
SUITEERROR=0

rapids-logger "Check GPU usage"
nvidia-smi

set +e

rapids-logger "Running googletests"
for gt in "$CONDA_PREFIX/bin/gtests/librmm/"* ; do
    ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR}/
    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

exit ${SUITEERROR}
