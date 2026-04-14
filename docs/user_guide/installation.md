# Installation

This guide covers installing RMM. For general RAPIDS installation instructions, which includes RMM, see the [RAPIDS Installation Guide](https://docs.rapids.ai/install/).

## System Requirements

See the [RAPIDS Platform Support](https://docs.rapids.ai/platform-support/) for supported operating systems, CUDA versions, GPU architectures, and Python versions for each release.

## Installing with conda

The easiest way to install RMM and all of its dependencies is using conda. You can get a minimal conda installation with [miniforge](https://conda-forge.org/download/).

### Stable Release

Install the latest stable release:

```bash
conda install -c rapidsai -c conda-forge rmm cuda-version=13
```

The `cuda-version` metapackage selects the CUDA Toolkit major version, and requires a CUDA driver to be installed from that major version or newer.

### Nightly Builds

For the latest development version, install from the nightly channel:

```bash
conda install -c rapidsai-nightly -c conda-forge rmm cuda-version=13
```

Nightly builds are created from the `main` branch and may contain unreleased features or bug fixes. They provide no stability guarantees.

## Installing with pip

RMM can also be installed using pip. The CUDA driver must already be installed on your system.

```bash
pip install rmm-cu13  # For CUDA 13
# or
pip install rmm-cu12  # For CUDA 12
```

## Building from Source

Building from source gives you the latest features and allows you to customize the build.

### Clone and Create Development Environment

The conda environment files in `conda/environments/` pin all build prerequisites (compiler, CUDA toolkit, CMake, etc.) to known-good versions:

```bash
git clone https://github.com/rapidsai/rmm.git
cd rmm

# Create environment for CUDA 13
conda env create --name rmm_dev --file conda/environments/all_cuda-131_arch-$(uname -m).yaml
conda activate rmm_dev
```

### Build Using build.sh

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

# Link your target with RMM
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
  VERSION 26.06
  GITHUB_REPOSITORY rapidsai/rmm
  GIT_TAG main
  SOURCE_SUBDIR cpp
)

target_link_libraries(your_target PRIVATE rmm::rmm)
```

## Testing Installation

### C++

Create a test file `test_rmm.cpp`:

```{literalinclude} ../../cpp/examples/docs/src/installation.cpp
---
language: cpp
start-after: "// [test-installation]"
end-before: "// [/test-installation]"
dedent:
---
```

Compile and run:

```bash
nvcc -std=c++17 -I/path/to/rmm/include test_rmm.cpp -o test_rmm
./test_rmm
```

### Python

```{literalinclude} ../../python/rmm/rmm/tests/examples/installation.py
---
language: python
start-after: "# [test-installation]"
end-before: "# [/test-installation]"
dedent:
---
```
