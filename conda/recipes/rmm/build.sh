# Copyright (c) 2018-2019, NVIDIA CORPORATION.

# Script assumes the script is executed from the root of the repo directory
printenv
git clean -xdf
./build.sh -v clean rmm
