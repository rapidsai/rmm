set -e

echo "Building librmm"
CUDA_REL=${CUDA_VERSION%.*}

conda build conda/recipes/librmm --python=$PYTHON
