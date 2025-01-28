#!/bin/bash

./build.sh -n -v clean librmm tests benchmarks --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
cmake --install build --component testing
