# Installation

This guide covers installing RMM. For general RAPIDS installation instructions, which includes RMM, see the [RAPIDS Installation Guide](https://docs.rapids.ai/install/).

## System Requirements

- **Operating System**: Linux or Windows Subsystem for Linux 2 (WSL2)
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **CUDA**: 12.2 or later
- **GPU**: Volta architecture or newer (Compute Capability 7.0+)

## Installing with conda

The easiest way to install RMM and all of its dependencies is using conda. You can get a minimal conda installation with [miniforge](https://conda-forge.org/download/).

### Stable Release

Install the latest stable release:

```bash
conda install -c rapidsai -c conda-forge rmm cuda-version=13
```

### Nightly Builds

For the latest development version, install from the nightly channel:

```bash
conda install -c rapidsai-nightly -c conda-forge rmm cuda-version=13
```

Nightly builds are created from the `main` branch and may contain unreleased features or bug fixes.

## Installing with pip

RMM can also be installed using pip, but requires that CUDA is already installed on your system.

```bash
pip install rmm-cu13  # For CUDA 13
# or
pip install rmm-cu12  # For CUDA 12
```

## Building from Source

Building from source gives you the latest features and allows you to customize the build.

### Development Environment

For a complete development environment, you can create an environment with all dependencies:

```bash
# Clone the repository
git clone https://github.com/rapidsai/rmm.git
cd rmm

# Create environment for CUDA 13
conda env create --name rmm_env --file conda/environments/all_cuda-130_arch-$(uname -m).yaml

# Activate the environment
conda activate rmm_env
```

### Prerequisites

- **GCC**: 13 or later
- **nvcc**: CUDA 12.2 or later
- **CMake**: 3.30.4 or later

### Build Steps

#### Clone the Repository

```bash
git clone https://github.com/rapidsai/rmm.git
cd rmm
```

#### Create Conda Development Environment

```bash
# For CUDA 13
conda env create --name rmm_dev --file conda/environments/all_cuda-130_arch-$(uname -m).yaml

# Activate the environment
conda activate rmm_dev
```

#### Build Using build.sh

RMM provides a convenience script `build.sh` that handles the build process.
The `build.sh` script is meant to be used with the developer conda environment above, which installs all prerequisites.

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

### Using CPM to Fetch RMM

You can use CPM to fetch RMM as a dependency:

```cmake
include(CPM)

CPMAddPackage(
  NAME rmm
  VERSION 26.02
  GITHUB_REPOSITORY rapidsai/rmm
  GIT_TAG main
  SOURCE_SUBDIR cpp
)

target_link_libraries(your_target PRIVATE rmm::rmm)
```

## Testing Installation

### C++

Create a test file `test_rmm.cpp`:

```cpp
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
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

### Python

```python
import rmm
print(rmm.__version__)

# Quick test
buffer = rmm.DeviceBuffer(size=100)
print(f"Allocated {buffer.size} bytes")
```
