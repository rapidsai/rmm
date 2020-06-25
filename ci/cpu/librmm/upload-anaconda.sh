#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

if [ "$UPLOAD_LIBRMM" == '1' ]; then
  export UPLOADFILE=$(conda build conda/recipes/librmm --python=$PYTHON --output)

  SOURCE_BRANCH=master
  CUDA_REL=${CUDA_VERSION%.*}

  SOURCE_BRANCH=master

  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"

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
  tar czvf librmm.tgz $CONDA_PREFIX/conda-bld/librmm*/work /$CONDA_PREFIX/conda-bld/{noarch,linux-64} $CONDA_PREFIX/conda-bld/channeldata.json
  export AWS_DEFAULT_REGION="us-east-2"
  logger "Upload conda pkg for librmm..."
  aws s3 cp librmm.tgz s3://gpuci-cache/rapidsai/rmm/${FLASH_ID}/librmm-${CUDA}.tgz
fi