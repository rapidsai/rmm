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

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --generate conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" > env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  rmm librmm

TESTRESULTS_DIR="${PWD}/test-results"
mkdir -p "${TESTRESULTS_DIR}"
SUITEERROR=0

rapids-logger "Check GPU usage"
nvidia-smi

cd python

set +e

rapids-logger "pytest rmm"
pytest --cache-clear --junitxml="${TESTRESULTS_DIR}/junit-rmm.xml" -v --cov-config=.coveragerc --cov=rmm --cov-report=xml:python/rmm-coverage.xml --cov-report term
exitcode=$?
if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in /rmm/python"
fi

exit ${SUITEERROR}
