# Copyright (c) 2018-2019, NVIDIA CORPORATION.

# Script assumes the script is executed from the root of the repo directory
BUILD_FLAGS=""
if [ "${cudaMallocAsync}" == "no_cma" ]; then
    BUILD_FLAGS="${BUILD_FLAGS} --no-cudamallocasync"
fi

./build.sh -v clean rmm ${BUILD_FLAGS}
