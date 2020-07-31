set -e

echo "Building rmm"
export RMM_BUILD_NO_GPU_TEST=1
if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    conda build conda/recipes/rmm --python=$PYTHON
else
    conda build -c ci/artifacts/rmm/cpu/conda-bld/ --dirty --no-remove-work-dir --python=$PYTHON conda/recipes/rmm
fi
