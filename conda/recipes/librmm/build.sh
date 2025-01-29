# Copyright (c) 2018-2025, NVIDIA CORPORATION.

./build.sh -n -v clean librmm tests benchmarks --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib -GNinja\"
