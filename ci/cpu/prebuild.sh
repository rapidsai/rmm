#!/usr/bin/env bash

#Build rmm once per PYTHON
if [[ "$CUDA" == "9.2" ]]; then
    export BUILD_RMM=1
else
    export BUILD_RMM=0
fi

#Build librmm once per CUDA
if [[ "$PYTHON" == "3.6" ]]; then
    export BUILD_LIBRMM=1
else
    export BUILD_LIBRMM=0
fi