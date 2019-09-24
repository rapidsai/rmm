set -e

if [ "$BUILD_LIBRMM" == '1' ]; then
  echo "Building librmm"
  CUDA_REL=${CUDA_VERSION%.*}

  conda build conda/recipes/librmm -c nvidia/label/cuda${CUDA_REL} -c rapidsai/label/cuda${CUDA_REL} -c rapidsai-nightly/label/cuda${CUDA_REL} -c conda-forge --python=$PYTHON
fi
