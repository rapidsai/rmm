# Basic Standalone librmm CUDA C++ application

This C++ example demonstrates a basic librmm use case and provides a minimal
example of building your own application based on librmm using CMake.

The example source code creates a device memory resource, sets it to the
current device resource, and then uses it to allocate a buffer. The buffer is
initialized with data and then deallocated.

## Compile and execute

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute
build/basic_example
```
