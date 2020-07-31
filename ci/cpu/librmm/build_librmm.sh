set -e

echo "Building librmm"
CUDA_REL=${CUDA_VERSION%.*}
if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    conda build conda/recipes/librmm --python=$PYTHON
else
    conda build --dirty --no-remove-work-dir conda/recipes/librmm
fi
