#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

# Setup 'gpuci_retry' for upload retries (results in 4 total attempts)
export GPUCI_RETRY_MAX=3
export GPUCI_RETRY_SLEEP=30

# Set default label options if they are not defined elsewhere
export LABEL_OPTION=${LABEL_OPTION:-"--label main"}

# Skip uploads unless BUILD_MODE == "branch"
if [ ${BUILD_MODE} != "branch" ]; then
  echo "Skipping upload"
  return 0
fi

# Skip uploads if there is no upload key
if [ -z "$MY_UPLOAD_KEY" ]; then
  echo "No upload key"
  return 0
fi


################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Starting conda uploads"

if [[ "$BUILD_LIBRMM" == "1" && "$UPLOAD_LIBRMM" == "1" ]]; then
  export LIBRMM_FILES=$(conda build --croot ${CONDA_BLD_DIR} conda/recipes/librmm --output)
  while read -r LIBRMM_FILE; do
    test -e ${LIBRMM_FILE}
    echo "Upload librmm file: ${LIBRMM_FILE}"
    gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${LIBRMM_FILE} --no-progress
  done <<< "${LIBRMM_FILES}"
fi

if [[ "$BUILD_RMM" == "1" && "$UPLOAD_RMM" == "1" ]]; then
  export RMM_FILES=$(conda build --croot ${CONDA_BLD_DIR} conda/recipes/rmm --python=$PYTHON --output)
  while read -r RMM_FILE; do
    test -e ${RMM_FILE}
    echo "Upload rmm file: ${RMM_FILE}"
    gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${RMM_FILE} --no-progress
  done <<< "${RMM_FILES}"
fi

