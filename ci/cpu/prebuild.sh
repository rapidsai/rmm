#!/usr/bin/env bash

#Always upload RMM packages
export UPLOAD_RMM=1
export UPLOAD_LIBRMM=1

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    #If project flash is not activate, always build both
    export BUILD_LIBRMM=1
    export BUILD_RMM=1
fi
