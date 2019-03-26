CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX11_ABI=$BUILD_ABI"

# show environment
printenv
# Cleanup local git
git clean -xdff
# Use CMake-based build procedure
mkdir build
cd build
# configure
cmake $CMAKE_COMMON_VARIABLES ..
# build
make -j${PARALLEL_LEVEL} VERBOSE=1 install
