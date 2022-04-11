#!/bin/bash
################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Get env"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Install build env
gpuci_mamba_retry install -y \
                  "cudatoolkit=11.0" \
                  "rapids-build-env=22.06.*"

# https://docs.rapids.ai/maintainers/depmgmt/
# conda remove --force rapids-build-env
# gpuci_mamba_retry install "your-pkg=1.0.0"

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls


CPP_FILE_NAME="conda_librmm_build_${BRANCH_NAME}-arc-${ARC}.tar"
PY_FILE_NAME="conda_rmm_build_${BRANCH_NAME}-arc-${ARC}.tar"

aws s3 cp "s3://rapids-downloads/ci/${CPP_FILE_NAME}" conda_cpp.tar
aws s3 cp "s3://rapids-downloads/ci/${PY_FILE_NAME}" conda_py.tar

mkdir cpp_artifact && tar -xvf conda_cpp.tar -C cpp_artifact/
mkdir py_artifact && tar -xvf conda_py.tar -C py_artifact/

gpuci_mamba_retry install -c cpp_artifact/.conda-bld -c py_artifact/.py-conda-bld rmm librmm

TESTRESULTS_DIR=${WORKSPACE}/test-results
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

gpuci_logger "Check GPU usage"
nvidia-smi

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

cd $WORKSPACE/python
export LIBRMM_BUILD_DIR="$WORKSPACE/ci/artifacts/rmm/cpu/conda_work/build"


gpuci_logger "pytest rmm"
py.test --cache-clear --junitxml=${WORKSPACE}/test-results/junit-rmm.xml -v --cov-config=.coveragerc --cov=rmm --cov-report=xml:${WORKSPACE}/python/rmm-coverage.xml --cov-report term
exitcode=$?
if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in /rmm/python"
fi

exit ${SUITEERROR}