#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

if [ "$UPLOAD_RMM" == "1" ]; then
  export UPLOADFILE=$(conda build conda/recipes/rmm --python=$PYTHON --output)

  SOURCE_BRANCH=master

  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${UPLOADFILE}

  # Restrict uploads to master branch
  if [ ${GIT_BRANCH} != ${SOURCE_BRANCH} ]; then
    echo "Skipping upload"
    return 0
  fi

  if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    return 0
  fi

  echo "Upload"
  echo ${UPLOADFILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${UPLOADFILE}
fi

if [[ "$PROJECT_FLASH" == "1" ]]; then
  tar czvf rmm.tgz $CONDA_PREFIX/conda-bld/rmm*/work /$CONDA_PREFIX/conda-bld/{noarch,linux-64} $CONDA_PREFIX/conda-bld/channeldata.json
  export AWS_DEFAULT_REGION="us-east-2"
  logger "Upload conda pkg for rmm..."
  aws s3 cp rmm.tgz s3://gpuci-cache/rapidsai/rmm/${FLASH_ID}/rmm-${CUDA}.tgz
fi