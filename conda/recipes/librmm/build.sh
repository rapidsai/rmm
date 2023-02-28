# Copyright (c) 2018-2022, NVIDIA CORPORATION.

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin ${CXX}" # Needed for CUDA 12 nvidia channel compilers
./build.sh -n -v clean librmm tests benchmarks --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
