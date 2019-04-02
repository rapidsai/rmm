set -e

if [ "$BUILD_RMM" == "1" ]; then
  echo "Building rmm"
  export RMM_BUILD_NO_GPU_TEST=1

  conda build conda/recipes/rmm -c rapidsai -c rapidsai-nightly -c nvidia -c conda-forge --python=$PYTHON
fi
