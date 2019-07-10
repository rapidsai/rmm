CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release"


# Cleanup local git
git clean -xdff
# Use CMake-based build procedure
mkdir build
cd build
# configure
cmake $CMAKE_COMMON_VARIABLES ..
# build
make -j${PARALLEL_LEVEL} VERBOSE=1 rmm_python_cffi
# install
make -j${PARALLEL_LEVEL} VERBOSE=1 rmm_install_python
