# Installation

This guide covers installing RMM using conda or building from source.

## System Requirements

- **Operating System**: Linux (RMM is only supported and tested on Linux)
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **CUDA**: 12.0 or later
- **GPU**: Volta architecture or newer (Compute Capability 7.0+)

## Installing with Conda (Recommended)

The easiest way to install RMM is using conda. You can get a minimal conda installation with [miniforge](https://github.com/conda-forge/miniforge).

### Stable Release

Install the latest stable release:

```bash
conda install -c rapidsai -c conda-forge -c nvidia rmm cuda-version=13.0
```

Replace `cuda-version=13.0` with your CUDA version (e.g., `cuda-version=12.0` for CUDA 12.0).

### Nightly Builds

For the latest development version, install from the nightly channel:

```bash
conda install -c rapidsai-nightly -c conda-forge -c nvidia rmm cuda-version=13.0
```

Nightly builds are created from the HEAD of the development branch and may contain unreleased features or bug fixes.

### Conda Environment

For a complete development environment, you can create an environment with all dependencies:

```bash
# Clone the repository
git clone https://github.com/rapidsai/rmm.git
cd rmm

# Create environment for CUDA 13.0 on x86_64
conda env create --name rmm_env --file conda/environments/all_cuda-130_arch-x86_64.yaml

# Activate the environment
conda activate rmm_env
```

Replace `x86_64` with `aarch64` for ARM architectures, and `130` with your CUDA version.

## Installing with pip

RMM can also be installed using pip, but requires that CUDA is already installed on your system.

```bash
pip install rmm-cu12  # For CUDA 12.x
# or
pip install rmm-cu11  # For CUDA 11.x
```

## Building from Source

Building from source gives you the latest features and allows you to customize the build.

### Prerequisites

#### Compilers

- **GCC**: 9.3 or later
- **nvcc**: CUDA 12.0 or later (part of CUDA Toolkit)
- **CMake**: 3.30.4 or later

#### CUDA Toolkit

CUDA 12.0 or later is required. Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).

#### Python Dependencies

The following Python packages are required (installed automatically when using conda environments):

- `rapids-build-backend` (from PyPI or rapidsai conda channel)
- `scikit-build-core`
- `cuda-python`
- `cython`

See [pyproject.toml](https://github.com/rapidsai/rmm/blob/main/python/rmm/pyproject.toml) for complete details.

### Build Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/rapidsai/rmm.git
cd rmm
```

#### 2. Create Conda Development Environment

```bash
# For CUDA 13.0 on x86_64
conda env create --name rmm_dev --file conda/environments/all_cuda-130_arch-$(arch).yaml

# Activate the environment
conda activate rmm_dev
```

#### 3. Build C++ Library (librmm)

Using CMake directly:

```bash
mkdir build
cd build

# Configure - use $CONDA_PREFIX as install path if using conda
cmake .. -DCMAKE_INSTALL_PREFIX=/install/path

# Build - uses all available CPU cores
make -j

# Install
make install
```

The `nvcc` executable must be on your `PATH` or defined in the `CUDACXX` environment variable:

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
```

#### 4. Build Using build.sh (Alternative)

RMM provides a convenience script that handles the build process:

```bash
# Show help
./build.sh -h

# Build librmm without installing
./build.sh -n librmm

# Build rmm Python package without installing
./build.sh -n rmm

# Build and install both
./build.sh librmm rmm
```

The `build.sh` script creates a `build` directory at the repository root.

#### 5. Build Python Package

From the repository root:

```bash
# Build the Python package
python -m pip wheel ./python/librmm
python -m pip install --find-links=. -e ./python/rmm
```

#### 6. Run Tests (Optional)

C++ tests:

```bash
cd build
make test
```

Python tests:

```bash
pytest -v
```

### CMake Build Options

RMM provides several CMake options to customize the build:

```bash
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/install/path \
  -DCMAKE_BUILD_TYPE=Release \
  -DRMM_LOGGING_LEVEL=INFO \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=OFF
```

#### Common Options

- `CMAKE_INSTALL_PREFIX`: Installation directory (default: `/usr/local`)
- `CMAKE_BUILD_TYPE`: Build type - `Release`, `Debug`, `RelWithDebInfo` (default: `Release`)
- `BUILD_TESTS`: Build C++ tests (default: `ON`)
- `BUILD_BENCHMARKS`: Build benchmarks (default: `OFF`)
- `RMM_LOGGING_LEVEL`: Logging verbosity - `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`, `OFF` (default: `INFO`)

### Caching Third-Party Dependencies

RMM uses [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) to manage third-party dependencies (CCCL, nvbench, etc.). By default, CPM downloads dependencies as needed.

To avoid repeated downloads when building frequently, set the `CPM_SOURCE_CACHE` environment variable:

```bash
export CPM_SOURCE_CACHE=$HOME/.cache/cpm
```

CPM will cache downloaded dependencies in this directory and reuse them across builds.

## Using RMM in a Downstream CMake Project

To use RMM in your own CMake project, add the following to your `CMakeLists.txt`:

```cmake
find_package(rmm REQUIRED)

# Link your target with RMM (header-only, pulls in dependencies)
target_link_libraries(your_target PRIVATE rmm::rmm)
```

If RMM is not installed in a default location, specify its path:

```bash
cmake .. -Drmm_ROOT=/path/to/rmm/install
```

Or use `CMAKE_PREFIX_PATH`:

```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/rmm/install
```

### Customizing Thrust

RMM depends on Thrust (part of CCCL), which is automatically pulled in via the `rmm::Thrust` target. To customize Thrust configuration:

```cmake
set(THRUST_HOST_SYSTEM CPP)
set(THRUST_DEVICE_SYSTEM CUDA)

find_package(rmm REQUIRED)
target_link_libraries(your_target PRIVATE rmm::rmm)
```

### Using CPM to Fetch RMM

You can use CPM to fetch RMM as a dependency:

```cmake
include(CPM)

CPMAddPackage(
  NAME rmm
  VERSION 25.12
  GITHUB_REPOSITORY rapidsai/rmm
  GIT_TAG main
  SYSTEM Off  # Important: prevents CCCL headers from being marked as SYSTEM
)

target_link_libraries(your_target PRIVATE rmm::rmm)
```

**Important**: Use CPM's multi-argument syntax with `SYSTEM Off`. The single-argument compact syntax marks dependencies as `SYSTEM`, which causes outdated CCCL headers from the CUDA SDK to take precedence over the correct versions pulled by CPM.

## Verifying Installation

### Python

```python
import rmm
print(rmm.__version__)

# Quick test
buffer = rmm.DeviceBuffer(size=100)
print(f"Allocated {buffer.size} bytes")
```

### C++

Create a test file `test_rmm.cpp`:

```cpp
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <iostream>

int main() {
    auto mr = rmm::mr::cuda_memory_resource{};
    rmm::mr::set_current_device_resource(&mr);

    rmm::device_buffer buf(100);
    std::cout << "Allocated " << buf.size() << " bytes\n";

    return 0;
}
```

Compile and run:

```bash
nvcc -std=c++17 -I/path/to/rmm/include test_rmm.cpp -o test_rmm
./test_rmm
```

## Troubleshooting

### GCC Version Issues

RMM requires GCC 9.3 or later. If you have an older version:

```bash
# Check your GCC version
gcc --version

# Install a newer GCC (Ubuntu/Debian)
sudo apt install gcc-11 g++-11

# Set as default (optional)
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```

### CUDA Not Found

Ensure CUDA is installed and `nvcc` is on your PATH:

```bash
# Check CUDA installation
nvcc --version

# Add CUDA to PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CMake Version

RMM requires CMake 3.30.4 or later:

```bash
# Check CMake version
cmake --version

# Install newer CMake from pip
pip install cmake

# Or download from https://cmake.org/download/
```

### Python Package Not Found

If `import rmm` fails after building from source:

```bash
# Ensure you installed the Python package
python -m pip install -e ./python/rmm

# Check installation
python -c "import rmm; print(rmm.__file__)"
```
