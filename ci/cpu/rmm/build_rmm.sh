set -e

echo "Building rmm"
export RMM_BUILD_NO_GPU_TEST=1

conda build conda/recipes/rmm --python=$PYTHON
