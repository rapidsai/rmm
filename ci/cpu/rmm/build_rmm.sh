set -e

echo "Building rmm"
export RMM_BUILD_NO_GPU_TEST=1
if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    conda build conda/recipes/rmm --python=$PYTHON
else
    git clean -xdf
    logger "Install librmm from s3"
    export AWS_DEFAULT_REGION="us-east-2"
    mkdir artifacts
    cd artifacts
    aws s3 cp s3://gpuci-cache/rapidsai/rmm/test/librmm-${CUDA}.tgz librmm.tgz
    tar xf librmm.tgz
    cd ..

    conda build -c artifacts/${CONDA_PREFIX}/conda-bld/ --dirty --no-remove-work-dir --variants "{python: [3.6, 3.7]}" conda/recipes/rmm
fi
