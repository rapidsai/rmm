# Copyright (c) 2018-2019, NVIDIA CORPORATION.

# Script assumes the script is executed from the root of the repo directory
printenv
if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    git clean -xdf
fi
./build.sh -v clean rmm
