#!/bin/bash

set -euo pipefail

# cat <<< '
# auto_update_conda: False
# ssl_verify: False
# channels:
#   - gpuci
#   - rapidsai-nightly
#   - dask/label/dev
#   - rapidsai
#   - pytorch
#   - conda-forge
#   - nvidia
# conda-build:
#   set_build_id: false
#   root_dir: /tmp/conda-bld-workspace
#   output_folder: /tmp/conda-bld-output
# ' > /opt/conda/.condarc

. /opt/conda/etc/profile.d/conda.sh
conda activate base

rapids-dependency-file-generator \
  --generate conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*}" > env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  librmm librmm-tests

TESTRESULTS_DIR=test-results
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

rapids-logger "Check GPU usage"
nvidia-smi

set +e

rapids-logger "Running googletests"
for gt in "$CONDA_PREFIX/bin/gtests/librmm/"* ; do
    ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

exit ${SUITEERROR}
