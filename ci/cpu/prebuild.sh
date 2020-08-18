#!/usr/bin/env bash

#Always upload RMM Python package
export UPLOAD_RMM=1

#Build librmm once per CUDA
if [[ "$PYTHON" == "3.7" ]]; then
    export UPLOAD_LIBRMM=1
else
    export UPLOAD_LIBRMM=0
fi

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    #If project flash is not activate, always build both
    export BUILD_LIBRMM=1
    export BUILD_RMM=1
fi
