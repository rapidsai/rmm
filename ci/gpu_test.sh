#!/bin/bash
set -e

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Get env"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

echo "CUDA version for mamba install: ${CUDA}"

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls


CPP_FILE_NAME="conda_librmm_build_${BRANCH_NAME}-arc-${ARC}.tar"
PY_FILE_NAME="conda_rmm_build_${BRANCH_NAME}-arc-${ARC}-py-${PY_VER}.tar"

aws s3 cp "s3://rapids-downloads/ci/${CPP_FILE_NAME}" conda_cpp.tar
aws s3 cp "s3://rapids-downloads/ci/${PY_FILE_NAME}" conda_py.tar
mkdir cpp__artifact py__artifact
tar -xvf conda_cpp.tar -C cpp__artifact/
tar -xvf conda_py.tar -C py__artifact/

gpuci_mamba_retry install -y \
    -c ./cpp__artifact/.conda-bld \
    -c ./py__artifact/.py-conda-bld \
    rmm librmm librmm-tests

TESTRESULTS_DIR=${WORKSPACE}/test-results
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

gpuci_logger "Check GPU usage"
nvidia-smi

set +e
gpuci_logger "Running googletests"
# run gtests from librmm-tests package
for gt in "$CONDA_PREFIX/bin/gtests/librmm/"* ; do
    ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

cd python

gpuci_logger "pytest rmm"
py.test --cache-clear --junitxml=test-results/junit-rmm.xml -v --cov-config=.coveragerc --cov=rmm --cov-report=xml:python/rmm-coverage.xml --cov-report term
exitcode=$?
if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in /rmm/python"
fi

exit ${SUITEERROR}
