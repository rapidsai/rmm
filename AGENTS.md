# AGENTS.md - RMM Development Guide

RMM (RAPIDS Memory Manager) is a CUDA memory management library providing C++ and Python APIs.

## Safety Rules for Agents

- **Minimal diffs**: Change only what's necessary; avoid drive-by refactors.
- **No mass reformatting**: Don't run formatters over unrelated code.
- **No API invention**: Align with existing RMM patterns and documented APIs.
- **Don't bypass CI**: Don't suggest skipping checks or using `--no-verify`.
- **CUDA/GPU hygiene**: Keep operations stream-ordered, use RMM allocators (never raw `new`/`delete` for device memory).

### Before Finalizing a Change

Ask yourself:
- What scenarios must be covered? (happy path, edge cases, failure modes)
- What's the expected behavior contract? (inputs/outputs, errors)
- Where should tests live? (C++ gtests under `cpp/tests/`, Python pytest under `python/rmm/rmm/tests/`)

## Build Commands

### Devcontainer (username: coder)
```bash
build-rmm-cpp -j0      # Build C++ library, accepts CMake flags (-D...)
build-rmm-python -j0   # Build Python package
build-rmm -j0          # Build both C++ and Python, accepts CMake flags (-D...)
```

### Standard Environment (requires conda env from conda/environments/)
```bash
./build.sh librmm              # Build and install C++ library
./build.sh librmm -g           # Debug build
./build.sh librmm tests        # Build with tests
./build.sh librmm benchmarks   # Build with benchmarks
./build.sh clean librmm        # Clean rebuild
./build.sh rmm                 # Build and install Python package
./build.sh librmm rmm          # Build both C++ and Python
```

### CMake Direct
```bash
cmake -S cpp -B cpp/build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j$(nproc)
```

## Test Commands

### Devcontainer (username: coder)
```bash
test-rmm-cpp      # Run all C++ tests, accepts ctest flags
test-rmm-python   # Run all Python tests, accepts pytest flags
```

### C++ Tests (GoogleTest)
```bash
# Run all C++ tests
ctest --test-dir cpp/build --output-on-failure

# Run single test by name pattern
ctest --test-dir cpp/build -R device_uvector

# Run specific test executable directly
./cpp/build/gtests/DEVICE_UVECTOR_TEST

# Run specific test case within executable
./cpp/build/gtests/DEVICE_UVECTOR_TEST --gtest_filter="TypedUVectorTest.*"
./cpp/build/gtests/DEVICE_UVECTOR_TEST --gtest_filter="*ZeroSizeConstructor"
```

### Python Tests (pytest)
```bash
# Run all Python tests
pytest python/rmm/rmm/tests/ -v

# Run single test file
pytest python/rmm/rmm/tests/test_rmm.py -v

# Run single test function
pytest python/rmm/rmm/tests/test_rmm.py::test_reinitialize -v

# Run with coverage
pytest python/rmm/rmm/tests/ --cov=rmm --cov-report=xml
```

## Lint and Format

Always use pre-commit to run linters and formatters:
```bash
pre-commit run --all-files     # Run all hooks (recommended)
pre-commit run clang-format --all-files  # C++ formatting only
pre-commit run ruff-check --all-files    # Python linting only
pre-commit run ruff-format --all-files   # Python formatting only
```

## Code Style Guidelines

### C++ Style
- **Standard**: C++20
- **Line length**: 100 characters
- **Indentation**: 2 spaces, no tabs
- **Braces**: WebKit style (same line for control statements)
- **Pointers**: Left-aligned (`int* ptr`, not `int *ptr`)
- **Namespaces**: No indentation inside namespaces
- **Include order** (enforced by clang-format):
  1. Quoted includes (`"local.hpp"`)
  2. RMM includes (`<rmm/...>`)
  3. CCCL includes (`<thrust/...>`, `<cub/...>`, `<cuda/...>`)
  4. CUDA includes (`<cuda_runtime.h>`)
  5. Other system includes (with `.` in name)
  6. STL includes (no `.` in name)

### Python Style
- **Line length**: 79 characters
- **Formatter**: ruff (replaces black, isort)
- **Type hints**: Required (enforced by mypy)
- **Imports**: Sorted by ruff/isort, grouped as stdlib → third-party → first-party
- **Docstrings**: NumPy style, triple double quotes

### Naming Conventions
- **C++ classes**: `snake_case` (e.g., `device_buffer`, `cuda_stream_view`)
- **C++ functions/methods**: `snake_case`
- **C++ constants/macros**: `SCREAMING_SNAKE_CASE`
- **C++ template params**: `PascalCase` with `T` suffix (e.g., `OffsetT`)
- **Python**: `snake_case` for functions, variables; `PascalCase` for classes

### Error Handling
- **C++**: Use RMM exception types from `<rmm/error.hpp>`
  - `rmm::bad_alloc` for allocation failures
  - `rmm::logic_error` for programming errors
  - Use `RMM_EXPECTS()` macro for precondition checks
  - Use `RMM_CUDA_TRY()` for CUDA API calls
- **Python**: Use `RMMError` or standard Python exceptions

### File Headers (SPDX format, required)
```cpp
// C++/CUDA
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
```
```python
# Python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
```

### Documentation
- **C++**: Doxygen comments for public APIs (`/** ... */`)
- **Python**: NumPy-style docstrings
- All public functions must be documented

## Project Structure
```
cpp/                    # C++ source code
├── include/rmm/        # Public headers (header-only library)
├── src/                # Implementation files
├── tests/              # GoogleTest tests
└── benchmarks/         # Google Benchmark benchmarks
python/rmm/             # Python package
├── rmm/                # Main module
│   ├── pylibrmm/       # Cython bindings
│   └── tests/          # pytest tests
ci/                     # CI scripts
```

## PR Requirements
- All tests must pass
- Pre-commit checks must pass
- Update documentation for API changes
- Add tests for new functionality

## Common Patterns

### Memory Resource Usage (C++)
```cpp
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

// Create a memory resource and set it as current
rmm::mr::cuda_async_memory_resource mr{};
rmm::mr::set_current_device_resource_ref(mr);

auto ref = rmm::mr::get_current_device_resource_ref();
rmm::device_buffer buf(size, stream, ref);
```

### Allocate/Deallocate (C++)
```cpp
// Stream-ordered (async) allocation - preferred for performance
void* ptr = mr.allocate(stream, bytes);
mr.deallocate(stream, ptr, bytes);

// Synchronous allocation - blocks until memory is ready on all streams
void* ptr = mr.allocate_sync(bytes);
mr.deallocate_sync(ptr, bytes);
```

### Stream Usage (C++)
```cpp
#include <rmm/cuda_stream_view.hpp>
void func(rmm::cuda_stream_view stream) {
  // Use stream for async operations
}
```

### Python Memory Resource
```python
import rmm

# Create a memory resource and set it as current
mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)

ref = rmm.mr.get_current_device_resource()
buf = rmm.DeviceBuffer(size=size)
```

## Memory Management Guidelines

- Never use raw CUDA memory APIs for device memory - use memory resources (except in memory resource implementations)
- Prefer `rmm::device_uvector<T>` for typed device memory
- Prefer `rmm::device_buffer` for untyped device memory
- All operations should be stream-ordered - accept `rmm::cuda_stream_view`
- Views (`*_view` suffix) are non-owning - don't manage their lifetime

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Main build script (never used in devcontainers) | `build.sh` |
| CMake configuration | `cpp/CMakeLists.txt` |
| C++ public headers | `cpp/include/rmm/` |
| Memory resources | `cpp/include/rmm/mr/` |
| Device containers | `cpp/include/rmm/device_uvector.hpp`, `device_buffer.hpp` |
| Stream utilities | `cpp/include/rmm/cuda_stream.hpp`, `cuda_stream_view.hpp` |
| Error handling | `cpp/include/rmm/error.hpp` |
| Python bindings | `python/rmm/rmm/` |
| C++ tests | `cpp/tests/` |
| Python tests | `python/rmm/rmm/tests/` |
| CI configuration | `ci/` |

## Resources

- **Documentation**: https://docs.rapids.ai/api/rmm/stable/
- **GitHub Issues**: https://github.com/rapidsai/rmm/issues
