#!/usr/bin/env bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# rmm GPU build & testscript for CI  #
######################################
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf

# Install build env
conda install rapids-build-env=${MINOR_VERSION}.*

# https://docs.rapids.ai/maintainers/depmgmt/ 
# conda remove -f rapids-build-env
# conda install "your-pkg=1.0.0"

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    ################################################################################
    # BUILD - Build and install librmm and rmm
    ################################################################################

    logger "Build and install librmm and rmm..."
    "$WORKSPACE/build.sh" -v --ptds clean librmm rmm

    ################################################################################
    # Test - librmm
    ################################################################################

    if hasArg --skip-tests; then
        logger "Skipping Tests..."
    else
        logger "Check GPU usage..."
        nvidia-smi

        logger "Running googletests..."

        cd "${WORKSPACE}/build"
        GTEST_OUTPUT="xml:${WORKSPACE}/test-results/" make -j${PARALLEL_LEVEL} test

        logger "Python py.test for librmm_cffi..."
        cd $WORKSPACE/python
        py.test --cache-clear --junitxml=${WORKSPACE}/test-results/junit-rmm.xml -v --cov-config=.coveragerc --cov=rmm --cov-report=xml:${WORKSPACE}/python/rmm-coverage.xml --cov-report term
    fi
else
    export LD_LIBRARY_PATH="$WORKSPACE/ci/artifacts/rmm/cpu/conda_work/build:$LD_LIBRARY_PATH"

    TESTRESULTS_DIR=${WORKSPACE}/test-results
    mkdir -p ${TESTRESULTS_DIR}
    SUITEERROR=0

    logger "Check GPU usage..."
    nvidia-smi

    logger "Running googletests..."
    # run gtests
    cd $WORKSPACE/ci/artifacts/rmm/cpu/conda_work
    for gt in "build/gtests/*" ; do
        ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
        exitcode=$?
        if (( ${exitcode} != 0 )); then
            SUITEERROR=${exitcode}
            echo "FAILED: GTest ${gt}"
        fi
    done

    cd $WORKSPACE/python
    
    logger "Installing librmm..."
    conda install -c $WORKSPACE/ci/artifacts/rmm/cpu/conda-bld/ librmm
    export LIBRMM_BUILD_DIR="$WORKSPACE/ci/artifacts/rmm/cpu/conda_work/build"
    
    logger "Building rmm"
    "$WORKSPACE/build.sh" -v rmm
    
    logger "pytest rmm"
    py.test --cache-clear --junitxml=${WORKSPACE}/test-results/junit-rmm.xml -v --cov-config=.coveragerc --cov=rmm --cov-report=xml:${WORKSPACE}/python/rmm-coverage.xml --cov-report term
    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: 1 or more tests in /rmm/python"
    fi

    exit ${SUITEERROR}
fi