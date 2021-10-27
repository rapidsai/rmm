#!/usr/bin/env bash
# Copyright (c) 2020, NVIDIA CORPORATION.
######################################
# rmm GPU build & testscript for CI  #
######################################
set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Determine CUDA release version
export CUDA_MAJOR_VER=$(echo "${CUDA_VERSION}" | cut -f 1 -d.)
export CUDA_MINOR_VER=$(echo "${CUDA_VERSION}" | cut -f 2 -d.)
export CUDA_REL="${CUDA_MAJOR_VER}.${CUDA_MINOR_VER}"

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
unset GIT_DESCRIBE_TAG

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Get env"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    # Install build env
    gpuci_mamba_retry install -y \
                  "cudatoolkit=$CUDA_REL" \
                  "rapids-build-env=${MINOR_VERSION}.*"

    # https://docs.rapids.ai/maintainers/depmgmt/
    # conda remove --force rapids-build-env
    # gpuci_mamba_retry install "your-pkg=1.0.0"

    ################################################################################
    # BUILD - Build and install librmm and rmm
    ################################################################################

    gpuci_logger "Build and install librmm and rmm"
    "$WORKSPACE/build.sh" -v clean librmm rmm benchmarks tests

    ################################################################################
    # Test - librmm
    ################################################################################

    if hasArg --skip-tests; then
        gpuci_logger "Skipping Tests"
    else
        gpuci_logger "Check GPU usage"
        nvidia-smi

        gpuci_logger "Running googletests"

        cd "${WORKSPACE}/build"
        GTEST_OUTPUT="xml:${WORKSPACE}/test-results/" make -j${PARALLEL_LEVEL} test

        gpuci_logger "Running rmm pytests..."
        cd $WORKSPACE/python
        py.test --cache-clear --basetemp=${WORKSPACE}/rmm-cuda-tmp --junitxml=${WORKSPACE}/test-results/junit-rmm.xml -v --cov-config=.coveragerc --cov=rmm --cov-report=xml:${WORKSPACE}/python/rmm-coverage.xml --cov-report term
    fi
else
    gpuci_mamba_retry install -c $WORKSPACE/ci/artifacts/rmm/cpu/.conda-bld/ librmm librmm-tests

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

    # TODO: Move boa install to gpuci/rapidsai
    gpuci_mamba_retry install boa
    export CONDA_BLD_DIR="$WORKSPACE/.conda-bld"
    gpuci_logger "Building and installing rmm"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} \
      -c $WORKSPACE/ci/artifacts/rmm/cpu/.conda-bld/ conda/recipes/rmm
    gpuci_mamba_retry install -c $WORKSPACE/ci/artifacts/rmm/cpu/.conda-bld/ \
      -c ${CONDA_BLD_DIR} rmm

    cd $WORKSPACE/python
    gpuci_logger "pytest rmm"
    py.test --cache-clear --junitxml=${WORKSPACE}/test-results/junit-rmm.xml -v --cov-config=.coveragerc --cov=rmm --cov-report=xml:${WORKSPACE}/python/rmm-coverage.xml --cov-report term
    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: 1 or more tests in /rmm/python"
    fi

    exit ${SUITEERROR}
fi
