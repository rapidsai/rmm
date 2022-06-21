#!/bin/bash

set -euo pipefail

# TODO: Move this
. /opt/conda/etc/profile.d/conda.sh
conda activate base

# Check environment
rapids-check-env

# GPU Test Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  rmm librmm librmm-tests

TESTRESULTS_DIR=test-results
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

rapids-logger "Check GPU usage"
nvidia-smi

set +e

rapids-logger "Running googletests"
# run gtests from librmm-tests package
for gt in "$CONDA_PREFIX/bin/gtests/librmm/"* ; do
    ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

exit ${SUITEERROR}
