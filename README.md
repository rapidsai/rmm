# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;RMM: RAPIDS Memory Manager</div>

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/rmm/blob/main/README.md) ensure you are on the `main` branch.

## Resources

- [RMM Reference Documentation](https://docs.rapids.ai/api/rmm/stable/): Python and C++ API references, tutorials, and topic guides.
- [RAPIDS Installation Guide](https://docs.rapids.ai/install/): Instructions for installing RMM.
- [GitHub Repository](https://github.com/rapidsai/rmm): Download the RMM source code.
- [Issue Tracker](https://github.com/rapidsai/rmm/issues): Report issues or request features.
- [RAPIDS Community](https://rapids.ai/learn-more/#get-involved): Get help, contribute, and collaborate.

## Overview

Achieving optimal performance in GPU-centric workflows frequently requires customizing how host and
device memory are allocated. For example, using "pinned" host memory for asynchronous
host <-> device memory transfers, or using a device memory pool sub-allocator to reduce the cost of
dynamic device memory allocation.

The goal of the RAPIDS Memory Manager (RMM) is to provide:
- A common interface that allows customizing [device](#device_memory_resource) and
  [host](#host_memory_resource) memory allocation
- A collection of [implementations](#available-resources) of the interface
- A collection of [data structures](#device-data-structures) that use the interface for memory allocation

For information on the interface RMM provides and how to use RMM in your C++ code, see
[below](#using-rmm-in-c).

For a walkthrough about the design of the RAPIDS Memory Manager, read [Fast, Flexible Allocation for NVIDIA CUDA with RAPIDS Memory Manager](https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/) on the NVIDIA Developer Blog.

## Installation

### Conda

RMM can be installed with conda. You can get a minimal conda installation with [miniforge](https://github.com/conda-forge/miniforge).

Install RMM with:

```bash
conda install -c rapidsai -c conda-forge -c nvidia rmm cuda-version=13.0
```

We also provide [nightly conda packages](https://anaconda.org/rapidsai-nightly) built from the HEAD
of our latest development branch.

Note: RMM is supported only on Linux, and only tested with Python versions 3.10, 3.11, 3.12, and 3.13.

Note: The RMM package from conda requires building with GCC 9 or later. Otherwise, your application may fail to build.

See the [RAPIDS Installation Guide](https://docs.rapids.ai/install/) for system requirements.

## Building from Source

### Get RMM Dependencies

Compiler requirements:

* `gcc`     version 9.3+
* `nvcc`    version 12.0+
* `cmake`   version 3.30.4+

CUDA/GPU requirements:

* CUDA 12.0+. You can obtain CUDA from
  [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

GPU Support:
* RMM is tested and supported only on Volta architecture and newer (Compute Capability 7.0+).

Python requirements:
* `rapids-build-backend` (available from PyPI or the `rapidsai` conda channel)
* `scikit-build-core`
* `cuda-python`
* `cython`

For more details, see [pyproject.toml](python/rmm/pyproject.toml)


### Script to build RMM from source

To install RMM from source, ensure the dependencies are met and follow the steps below:

- Clone the repository:

```bash
$ git clone https://github.com/rapidsai/rmm.git
$ cd rmm
```

- Create the conda development environment `rmm_dev`:

```bash
# create the conda environment (assuming in base `rmm` directory)
$ conda env create --name rmm_dev --file conda/environments/all_cuda-130_arch-$(arch).yaml
# activate the environment
$ conda activate rmm_dev
```

- Build and install `librmm` using cmake & make. CMake depends on the `nvcc` executable being on
  your path or defined in `CUDACXX` environment variable.

```bash
$ mkdir build                                       # make a build directory
$ cd build                                          # enter the build directory
$ cmake .. -DCMAKE_INSTALL_PREFIX=/install/path     # configure cmake ... use $CONDA_PREFIX if you're using Anaconda
$ make -j                                           # compile the library librmm.so ... '-j' will start a parallel job using the number of physical cores available on your system
$ make install                                      # install the library librmm.so to '/install/path'
```

- Building and installing `librmm` and `rmm` using build.sh. Build.sh creates build dir at root of
  git repository. build.sh depends on the `nvcc` executable being on your path or defined in
  `CUDACXX` environment variable.

```bash
$ ./build.sh -h                                     # Display help and exit
$ ./build.sh -n librmm                              # Build librmm without installing
$ ./build.sh -n rmm                                 # Build rmm without installing
$ ./build.sh -n librmm rmm                          # Build librmm and rmm without installing
$ ./build.sh librmm rmm                             # Build and install librmm and rmm
```

- To run tests (Optional):

```bash
$ cd build (if you are not already in build directory)
$ make test
```

- Build, install, and test the `rmm` python package, in the `python` folder:

```bash
# In the root rmm directory
$ python -m pip wheel ./python/librmm
$ python -m pip install --find-links=. -e ./python/rmm
$ pytest -v
```

Done! You are ready to develop for the RMM OSS project.

### Caching third-party dependencies

RMM uses [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) to
handle third-party dependencies like spdlog, Thrust, GoogleTest,
GoogleBenchmark. In general you won't have to worry about it. If CMake
finds an appropriate version on your system, it uses it (you can
help it along by setting `CMAKE_PREFIX_PATH` to point to the
installed location). Otherwise those dependencies will be downloaded as
part of the build.

If you frequently start new builds from scratch, consider setting the
environment variable `CPM_SOURCE_CACHE` to an external download
directory to avoid repeated downloads of the third-party dependencies.

## Using RMM in a downstream CMake project

The installed RMM library provides a set of config files that makes it easy to
integrate RMM into your own CMake project. Add the following to `CMakeLists.txt`:

```cmake
find_package(rmm [VERSION])
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) rmm::rmm)
```

Since RMM is a header-only library, this does not actually link RMM,
but it makes the headers available and pulls in transitive dependencies.
If RMM is not installed in a default location, use
`CMAKE_PREFIX_PATH` or `rmm_ROOT` to point to its location.

One of RMM's dependencies is the Thrust library, so the above
automatically pulls in `Thrust` by means of a dependency on the
`rmm::Thrust` target. By default it uses the standard configuration of
Thrust. If you want to customize it, you can set the variables
`THRUST_HOST_SYSTEM` and `THRUST_DEVICE_SYSTEM`; see
[Thrust's CMake documentation](https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/cmake/README.md).

### Using CPM to manage RMM

RMM uses [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) to manage
its dependencies, including [CCCL](https://github.com/nvidia/cccl), and you can
use CPM for your project's dependency on RMM.

There is an issue with using CPM's *single-argument compact syntax* for
RMM/CCCL as it transitively marks targets as `SYSTEM` dependencies.
This causes the CCCL headers pulled in through CPM to be of lower priority
to the preprocessor than the (potentially outdated) CCCL headers provided
by the CUDA SDK. To avoid this issue, use CPM's *multi-argument syntax*
instead:

```cmake
CPMAddPackage(NAME rmm [VERSION]
              GITHUB_REPOSITORY rapidsai/rmm
              SYSTEM Off)
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) rmm::rmm)
```

# Using RMM in C++

The first goal of RMM is to provide a common interface for device and host memory allocation.
This allows both _users_ and _implementers_ of custom allocation logic to program to a single
interface.

To this end, RMM defines two abstract interface classes:
- [`rmm::mr::device_memory_resource`](#device_memory_resource) for device memory allocation
- [`rmm::mr::host_memory_resource`](#host_memory_resource) for host memory allocation

These classes are based on the
[`std::pmr::memory_resource`](https://en.cppreference.com/w/cpp/memory/memory_resource) interface
class introduced in C++17 for polymorphic memory allocation.

## `device_memory_resource`

`rmm::mr::device_memory_resource` is the base class that defines the interface for allocating and
freeing device memory.

It has two key functions:

1. `void* device_memory_resource::allocate(std::size_t bytes, cuda_stream_view s)`
   - Returns a pointer to an allocation of at least `bytes` bytes.

2. `void device_memory_resource::deallocate(void* p, std::size_t bytes, cuda_stream_view s)`
   - Reclaims a previous allocation of size `bytes` pointed to by `p`.
   - `p` *must* have been returned by a previous call to `allocate(bytes)`, otherwise behavior is
     undefined

It is up to a derived class to provide implementations of these functions. See
[available resources](#available-resources) for example `device_memory_resource` derived classes.

Unlike `std::pmr::memory_resource`, `rmm::mr::device_memory_resource` does not allow specifying an
alignment argument. All allocations are required to be aligned to at least 256B. Furthermore,
`device_memory_resource` adds an additional `cuda_stream_view` argument to allow specifying the stream
on which to perform the (de)allocation.

## Stream-ordered Memory Allocation

`rmm::mr::device_memory_resource` is a base class that provides stream-ordered memory allocation.
This allows optimizations such as re-using memory deallocated on the same stream without the
overhead of synchronization.

A call to `device_memory_resource::allocate(bytes, stream_a)` returns a pointer that is valid to use
on `stream_a`. Using the memory on a different stream (say `stream_b`) is Undefined Behavior unless
the two streams are first synchronized, for example by using `cudaStreamSynchronize(stream_a)` or by
recording a CUDA event on `stream_a` and then calling `cudaStreamWaitEvent(stream_b, event)`.

The stream specified to `device_memory_resource::deallocate` should be a stream on which it is valid
to use the deallocated memory immediately for another allocation. Typically this is the stream
on which the allocation was *last* used before the call to `deallocate`. The passed stream may be
used internally by a `device_memory_resource` for managing available memory with minimal
synchronization, and it may also be synchronized at a later time, for example using a call to
`cudaStreamSynchronize()`.

For this reason, it is Undefined Behavior to destroy a CUDA stream that is passed to
`device_memory_resource::deallocate`. If the stream on which the allocation was last used has been
destroyed before calling `deallocate` or it is known that it will be destroyed, it is likely better
to synchronize the stream (before destroying it) and then pass a different stream to `deallocate`
(e.g. the default stream).

Note that device memory data structures such as `rmm::device_buffer` and `rmm::device_uvector`
follow these stream-ordered memory allocation semantics and rules.

For further information about stream-ordered memory allocation semantics, read
[Using the NVIDIA CUDA Stream-Ordered Memory
Allocator](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/)
on the NVIDIA Developer Blog.

## Available Device Resources

RMM provides several `device_memory_resource` derived classes to satisfy various user requirements.
For more detailed information about these resources, see their respective documentation.

### `cuda_memory_resource`

Allocates and frees device memory using `cudaMalloc` and `cudaFree`.

### `managed_memory_resource`

Allocates and frees device memory using `cudaMallocManaged` and `cudaFree`.

Note that `managed_memory_resource` cannot be used with NVIDIA Virtual GPU Software (vGPU, for use
with virtual machines or hypervisors) because [NVIDIA CUDA Unified Memory is not supported by
NVIDIA vGPU](https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html#cuda-open-cl-support-vgpu).

### `pool_memory_resource`

A coalescing, best-fit pool sub-allocator.

### `fixed_size_memory_resource`

A memory resource that can only allocate a single fixed size. Average allocation and deallocation
cost is constant.

### `binning_memory_resource`

Configurable to use multiple upstream memory resources for allocations that fall within different
bin sizes. Often configured with multiple bins backed by `fixed_size_memory_resource`s and a single
`pool_memory_resource` for allocations larger than the largest bin size.

## Default Resources and Per-device Resources

RMM users commonly need to configure a `device_memory_resource` object to use for all allocations
where another resource has not explicitly been provided. A common example is configuring a
`pool_memory_resource` to use for all allocations to get fast dynamic allocation.

To enable this use case, RMM provides the concept of a "default" `device_memory_resource`. This
resource is used when another is not explicitly provided.

Accessing and modifying the default resource is done through two functions:
- `device_memory_resource* get_current_device_resource()`
   - Returns a pointer to the default resource for the current CUDA device.
   - The initial default memory resource is an instance of `cuda_memory_resource`.
   - This function is thread safe with respect to concurrent calls to it and
     `set_current_device_resource()`.
   - For more explicit control, you can use `get_per_device_resource()`, which takes a device ID.

- `device_memory_resource* set_current_device_resource(device_memory_resource* new_mr)`
   - Updates the default memory resource pointer for the current CUDA device to `new_mr`
   - Returns the previous default resource pointer
   - If `new_mr` is `nullptr`, then resets the default resource to `cuda_memory_resource`
   - This function is thread safe with respect to concurrent calls to it and
     `get_current_device_resource()`
   - For more explicit control, you can use `set_per_device_resource()`, which takes a device ID.

### Example

```c++
rmm::mr::cuda_memory_resource cuda_mr;
// Construct a resource that uses a coalescing best-fit pool allocator
// With the pool initially half of available device memory
auto initial_size = rmm::percent_of_free_device_memory(50);
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr, initial_size};
rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`
rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(); // Points to `pool_mr`
```

### Multiple Devices

A `device_memory_resource` should only be used when the active CUDA device is the same device
that was active when the `device_memory_resource` was created. Otherwise behavior is undefined.

If a `device_memory_resource` is used with a stream associated with a different CUDA device than the
device for which the memory resource was created, behavior is undefined.

Creating a `device_memory_resource` for each device requires care to set the current device before
creating each resource, and to maintain the lifetime of the resources as long as they are set as
per-device resources. Here is an example loop that creates `unique_ptr`s to `pool_memory_resource`
objects for each device and sets them as the per-device resource for that device.

```c++
using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;
std::vector<unique_ptr<pool_mr>> per_device_pools;
for(int i = 0; i < N; ++i) {
  cudaSetDevice(i); // set device i before creating MR
  // Use a vector of unique_ptr to maintain the lifetime of the MRs
  // Note: for brevity, omitting creation of upstream and computing initial_size
  per_device_pools.push_back(std::make_unique<pool_mr>(upstream, initial_size));
  // Set the per-device resource for device i
  set_per_device_resource(cuda_device_id{i}, &per_device_pools.back());
}
```

Note that the CUDA device that is current when creating a `device_memory_resource` must also be
current any time that `device_memory_resource` is used to deallocate memory, including in a
destructor. The RAII class `rmm::device_buffer` and classes that use it as a backing store
(`rmm::device_scalar` and `rmm::device_uvector`) handle this by storing the active device when the
constructor is called, and then ensuring that the stored device is active whenever an allocation or
deallocation is performed (including in the destructor). The user must therefore only ensure that
the device active during _creation_ of an `rmm::device_buffer` matches the active device of the
memory resource being used.

Here is an incorrect example that creates a memory resource on device `0` and then uses it to
allocate a `device_buffer` on device `1`:

```c++
{
  RMM_CUDA_TRY(cudaSetDevice(0));
  auto mr = rmm::mr::cuda_memory_resource{};
  {
    RMM_CUDA_TRY(cudaSetDevice(1));
    // Invalid, current device is 1, but MR is only valid for device 0
    rmm::device_buffer buf(16, rmm::cuda_stream_default, &mr);
  }
}
```

A correct example creates the device buffer with device `0` active. After that it is safe to switch
devices and let the buffer go out of scope and destruct with a different device active. For example,
this code is correct:

```c++
{
  RMM_CUDA_TRY(cudaSetDevice(0));
  auto mr = rmm::mr::cuda_memory_resource{};
  rmm::device_buffer buf(16, rmm::cuda_stream_default, &mr);
  RMM_CUDA_TRY(cudaSetDevice(1));
  ...
  // No need to switch back to device 0 before ~buf runs
}
```

#### Use of `rmm::device_vector` with multiple devices

`rmm:device_vector` uses an `rmm::mr::thrust_allocator` to enable `thrust::device_vector` to
allocate and deallocate memory using RMM. As such, the usual rules for usage of the backing memory
resource apply: the active device must match the active device at resource construction time. To
facilitate use in an RAII setting, `rmm::mr::thrust_allocator` records the active device at
construction time and ensures that device is active whenever it allocates or deallocates memory.
Usage of `rmm::device_vector` with multiple devices is therefore the same as `rmm::device_buffer`.
One must _create_ `device_vector`s with the correct device active, but it is safe to destroy them
with a different active device.

For example, recapitulating the previous example using `rmm::device_vector`:

```c++
{
  RMM_CUDA_TRY(cudaSetDevice(0));
  auto mr = rmm::mr::cuda_memory_resource{};
  rmm::device_vector<int> vec(16, rmm::mr::thrust_allocator<int>(rmm::cuda_stream_default, &mr));
  RMM_CUDA_TRY(cudaSetDevice(1));
  ...
  // No need to switch back to device 0 before ~vec runs
}
```

> [!NOTE]
> Although allocation and deallocation in the `thrust_allocator` run with the correct active device,
> modification of `rmm::device_vector` might necessitate a kernel launch, and this must run with the
> correct device active. For example, `.resize()` might both allocate _and_ launch a kernel to
> initialize new elements: the user must arrange for this kernel launch to occur with the correct
> device for the memory resource active.

## `cuda_stream_view` and `cuda_stream`

`rmm::cuda_stream_view` is a simple non-owning wrapper around a CUDA `cudaStream_t`. This wrapper's
purpose is to provide strong type safety for stream types. (`cudaStream_t` is an alias for a pointer,
which can lead to ambiguity in APIs when it is assigned `0`.)  All RMM stream-ordered APIs take a
`rmm::cuda_stream_view` argument.

`rmm::cuda_stream` is a simple owning wrapper around a CUDA `cudaStream_t`. This class provides
RAII semantics (constructor creates the CUDA stream, destructor destroys it). An `rmm::cuda_stream`
can never represent the CUDA default stream or per-thread default stream; it only ever represents
a single non-default stream. `rmm::cuda_stream` cannot be copied, but can be moved.

## `cuda_stream_pool`

`rmm::cuda_stream_pool` provides fast access to a pool of CUDA streams. This class can be used to
create a set of `cuda_stream` objects whose lifetime is equal to the `cuda_stream_pool`. Using the
stream pool can be faster than creating the streams on the fly. The size of the pool is configurable.
Depending on this size, multiple calls to `cuda_stream_pool::get_stream()` may return instances of
`rmm::cuda_stream_view` that represent identical CUDA streams.

## Thread Safety

All current device memory resources are thread safe unless documented otherwise. More specifically,
calls to memory resource `allocate()` and `deallocate()` methods are safe with respect to calls to
either of these functions from other threads. They are _not_ thread safe with respect to
construction and destruction of the memory resource object.

Note that a class `thread_safe_resource_adapter` is provided which can be used to adapt a memory
resource that is not thread safe to be thread safe (as described above). This adapter is not needed
with any current RMM device memory resources.

## Allocators

C++ interfaces commonly allow customizable memory allocation through an [`Allocator`](https://en.cppreference.com/w/cpp/named_req/Allocator) object.
RMM provides several `Allocator` and `Allocator`-like classes.

### `polymorphic_allocator`

A [stream-ordered](#stream-ordered-memory-allocation) allocator similar to [`std::pmr::polymorphic_allocator`](https://en.cppreference.com/w/cpp/memory/polymorphic_allocator).
Unlike the standard C++ `Allocator` interface, the `allocate` and `deallocate` functions take a `cuda_stream_view` indicating the stream on which the (de)allocation occurs.

### `stream_allocator_adaptor`

`stream_allocator_adaptor` can be used to adapt a stream-ordered allocator to present a standard `Allocator` interface to consumers that may not be designed to work with a stream-ordered interface.

Example:
```c++
rmm::cuda_stream stream;
rmm::mr::polymorphic_allocator<int> stream_alloc;

// Constructs an adaptor that forwards all (de)allocations to `stream_alloc` on `stream`.
auto adapted = rmm::mr::stream_allocator_adaptor(stream_alloc, stream);

// Allocates 100 bytes using `stream_alloc` on `stream`
auto p = adapted.allocate(100);
...
// Deallocates using `stream_alloc` on `stream`
adapted.deallocate(p,100);
```

### `thrust_allocator`

`thrust_allocator` is a device memory allocator that uses the strongly typed `thrust::device_ptr`, making it usable with containers like `thrust::device_vector`.

See [below](#using-rmm-with-thrust) for more information on using RMM with Thrust.

## Device Data Structures

### `device_buffer`

An untyped, uninitialized RAII class for stream ordered device memory allocation.

#### Example

```c++
cuda_stream_view s{...};
// Allocates at least 100 bytes on stream `s` using the *default* resource
rmm::device_buffer b{100,s};
void* p = b.data();                   // Raw, untyped pointer to underlying device memory

kernel<<<..., s.value()>>>(b.data()); // `b` is only safe to use on `s`

rmm::mr::device_memory_resource * mr = new my_custom_resource{...};
// Allocates at least 100 bytes on stream `s` using the resource `mr`
rmm::device_buffer b2{100, s, mr};
```

### `device_uvector<T>`
A typed, uninitialized RAII class for allocation of a contiguous set of elements in device memory.
Similar to a `thrust::device_vector`, but as an optimization, does not default initialize the
contained elements. This optimization restricts the types `T` to trivially copyable types.

#### Example

```c++
cuda_stream_view s{...};
// Allocates uninitialized storage for 100 `int32_t` elements on stream `s` using the
// default resource
rmm::device_uvector<int32_t> v(100, s);
// Initializes the elements to 0
thrust::uninitialized_fill(thrust::cuda::par.on(s.value()), v.begin(), v.end(), int32_t{0});

rmm::mr::device_memory_resource * mr = new my_custom_resource{...};
// Allocates uninitialized storage for 100 `int32_t` elements on stream `s` using the resource `mr`
rmm::device_uvector<int32_t> v2{100, s, mr};
```

### `device_scalar`
A typed, RAII class for allocation of a single element in device memory.
This is similar to a `device_uvector` with a single element, but provides convenience functions like
modifying the value in device memory from the host, or retrieving the value from device to host.

#### Example
```c++
cuda_stream_view s{...};
// Allocates uninitialized storage for a single `int32_t` in device memory
rmm::device_scalar<int32_t> a{s};
a.set_value(42, s); // Updates the value in device memory to `42` on stream `s`

kernel<<<...,s.value()>>>(a.data()); // Pass raw pointer to underlying element in device memory

int32_t v = a.value(s); // Retrieves the value from device to host on stream `s`
```

## `host_memory_resource`

`rmm::mr::host_memory_resource` is the base class that defines the interface for allocating and
freeing host memory.

Similar to `device_memory_resource`, it has two key functions for (de)allocation:

1. `void* host_memory_resource::allocate(std::size_t bytes, std::size_t alignment)`
   - Returns a pointer to an allocation of at least `bytes` bytes aligned to the specified
     `alignment`

2. `void host_memory_resource::deallocate(void* p, std::size_t bytes, std::size_t alignment)`
   - Reclaims a previous allocation of size `bytes` pointed to by `p`.


Unlike `device_memory_resource`, the `host_memory_resource` interface and behavior is identical to
`std::pmr::memory_resource`.

## Available Host Resources

### `new_delete_resource`

Uses the global `operator new` and `operator delete` to allocate host memory.

### `pinned_memory_resource`

Allocates "pinned" host memory using `cuda(Malloc/Free)Host`.

## Host Data Structures

RMM does not currently provide any data structures that interface with `host_memory_resource`.
In the future, RMM will provide a similar host-side structure like `device_buffer` and an allocator
that can be used with STL containers.

## Using RMM with Thrust

RAPIDS and other CUDA libraries make heavy use of Thrust. Thrust uses CUDA device memory in two
situations:

 1. As the backing store for `thrust::device_vector`, and
 2. As temporary storage inside some algorithms, such as `thrust::sort`.

RMM provides `rmm::mr::thrust_allocator` as a conforming Thrust allocator that uses
`device_memory_resource`s.

### Thrust Algorithms

To instruct a Thrust algorithm to use `rmm::mr::thrust_allocator` to allocate temporary storage, you
can use the custom Thrust CUDA device execution policy: `rmm::exec_policy(stream)`.

```c++
thrust::sort(rmm::exec_policy(stream, ...);
```

The first `stream` argument is the `stream` to use for `rmm::mr::thrust_allocator`.
The second `stream` argument is what should be used to execute the Thrust algorithm.
These two arguments must be identical.

## Logging

RMM includes two forms of logging. Memory event logging and debug logging.

### Memory Event Logging and `logging_resource_adaptor`

Memory event logging writes details of every allocation or deallocation to a CSV (comma-separated
value) file. In C++, Memory Event Logging is enabled by using the `logging_resource_adaptor` as a
wrapper around any other `device_memory_resource` object.

Each row in the log represents either an allocation or a deallocation. The columns of the file are
"Thread, Time, Action, Pointer, Size, Stream".

The CSV output files of the `logging_resource_adaptor` can be used as input to `REPLAY_BENCHMARK`,
which is available when building RMM from source, in the `gbenchmarks` folder in the build directory.
This log replayer can be useful for profiling and debugging allocator issues.

The following C++ example creates a logging version of a `cuda_memory_resource` that outputs the log
to the file "logs/test1.csv".

```c++
std::string filename{"logs/test1.csv"};
rmm::mr::cuda_memory_resource upstream;
rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource> log_mr{&upstream, filename};
```

If a file name is not specified, the environment variable `RMM_LOG_FILE` is queried for the file
name. If `RMM_LOG_FILE` is not set, then an exception is thrown by the `logging_resource_adaptor`
constructor.

In Python, memory event logging is enabled when the `logging` parameter of `rmm.reinitialize()` is
set to `True`. The log file name can be set using the `log_file_name` parameter. See
`help(rmm.reinitialize)` for full details.

### Debug Logging

RMM leverages [`rapids-logger`](https://github.com/rapidsai/rapids-logger) to log trace and debug
information to a file. This information can show when errors occur, when additional memory is
allocated from upstream resources, etc. By default output is logged to stderr, but the environment
variable `RMM_DEBUG_LOG_FILE` can be set to specify a path and file name to dump the logs to
instead.

There is a CMake configuration variable `RMM_LOGGING_LEVEL`, which can be set to enable compilation
of more detailed logging. The default is `INFO`. Available levels are `TRACE`, `DEBUG`, `INFO`,
`WARN`, `ERROR`, `CRITICAL` and `OFF`.

Note that to see logging below the `INFO` level, the application must also set the logging level at
run time. C++ applications must must call `rmm::default_logger().set_level()`, for example to enable all
levels of logging down to `TRACE`, call `rmm::default_logger().set_level(rapids_logger::level_enum::trace)` (and compile
librmm with `-DRMM_LOGGING_LEVEL=TRACE`). Python applications must call `rmm.set_logging_level()`,
for example to enable all levels of logging down to `TRACE`, call `rmm.set_logging_level("trace")`
(and compile the RMM Python module with `-DRMM_LOGGING_LEVEL=TRACE`).

Note that debug logging is different from the CSV memory allocation logging provided by
`rmm::mr::logging_resource_adapter`. The latter is for logging a history of allocation /
deallocation actions which can be useful for replay with RMM's replay benchmark.

## RMM and CUDA Memory Bounds Checking

Memory allocations taken from a memory resource that allocates a pool of memory (such as
`pool_memory_resource` and `arena_memory_resource`) are part of the same low-level CUDA memory
allocation. Therefore, out-of-bounds or misaligned accesses to these allocations are not likely to
be detected by CUDA tools such as
[CUDA Compute Sanitizer](https://docs.nvidia.com/cuda/compute-sanitizer/index.html) memcheck.

Exceptions to this are `cuda_memory_resource`, which wraps `cudaMalloc`, and
`cuda_async_memory_resource`, which uses `cudaMallocAsync` with CUDA's built-in memory pool
functionality (introduced in CUDA 11.2). Illegal memory accesses to memory allocated by these
resources are detectable with Compute Sanitizer Memcheck.

It may be possible in the future to add support for memory bounds checking with other memory
resources using NVTX APIs.

# Using RMM in Python

There are two ways to use RMM in Python code:

1. Using the `rmm.DeviceBuffer` API to explicitly create and manage
   device memory allocations
2. Transparently via external libraries such as CuPy and Numba

RMM provides a `MemoryResource` abstraction to control _how_ device
memory is allocated in both the above uses.

## DeviceBuffer

A DeviceBuffer represents an **untyped, uninitialized device memory
allocation**.  DeviceBuffers can be created by providing the
size of the allocation in bytes:

```python
>>> import rmm
>>> buf = rmm.DeviceBuffer(size=100)
```

The size of the allocation and the memory address associated with it
can be accessed via the `.size` and `.ptr` attributes respectively:

```python
>>> buf.size
100
>>> buf.ptr
140202544726016
```

DeviceBuffers can also be created by copying data from host memory:

```python
>>> import rmm
>>> import numpy as np
>>> a = np.array([1, 2, 3], dtype='float64')
>>> buf = rmm.DeviceBuffer.to_device(a.tobytes())
>>> buf.size
24
```

Conversely, the data underlying a DeviceBuffer can be copied to the
host:

```python
>>> np.frombuffer(buf.tobytes())
array([1., 2., 3.])
```

## MemoryResource objects

`MemoryResource` objects are used to configure how device memory allocations are made by
RMM.

By default if a `MemoryResource` is not set explicitly, RMM uses the `CudaMemoryResource`, which
uses `cudaMalloc` for allocating device memory.

`rmm.reinitialize()` provides an easy way to initialize RMM with specific memory resource options
across multiple devices. See `help(rmm.reinitialize)` for full details.

For lower-level control, the `rmm.mr.set_current_device_resource()` function can be
used to set a different MemoryResource for the current CUDA device.  For
example, enabling the `ManagedMemoryResource` tells RMM to use
`cudaMallocManaged` instead of `cudaMalloc` for allocating memory:

```python
>>> import rmm
>>> rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
```

> :warning: The default resource must be set for any device **before**
> allocating any device memory on that device.  Setting or changing the
> resource after device allocations have been made can lead to unexpected
> behaviour or crashes. See [Multiple Devices](#multiple-devices)

As another example, `PoolMemoryResource` allows you to allocate a
large "pool" of device memory up-front. Subsequent allocations will
draw from this pool of already allocated memory.  The example
below shows how to construct a PoolMemoryResource with an initial size
of 1 GiB and a maximum size of 4 GiB. The pool uses
`CudaMemoryResource` as its underlying ("upstream") memory resource:

```python
>>> import rmm
>>> pool = rmm.mr.PoolMemoryResource(
...     rmm.mr.CudaMemoryResource(),
...     initial_pool_size="1GiB", # equivalent to initial_pool_size=2**30
...     maximum_pool_size="4GiB"
... )
>>> rmm.mr.set_current_device_resource(pool)
```
Other MemoryResources include:

* `FixedSizeMemoryResource` for allocating fixed blocks of memory
* `BinningMemoryResource` for allocating blocks within specified "bin" sizes from different memory
resources

MemoryResources are highly configurable and can be composed together in different ways.
See `help(rmm.mr)` for more information.

## Using RMM with third-party libraries

### Using RMM with CuPy

You can configure [CuPy](https://cupy.dev/) to use RMM for memory
allocations by setting the CuPy CUDA allocator to
`rmm_cupy_allocator`:

```python
>>> from rmm.allocators.cupy import rmm_cupy_allocator
>>> import cupy
>>> cupy.cuda.set_allocator(rmm_cupy_allocator)
```


**Note:** This only configures CuPy to use the current RMM resource for allocations.
It does not initialize nor change the current resource, e.g., enabling a memory pool.
See [here](#memoryresource-objects) for more information on changing the current memory resource.

### Using RMM with Numba

You can configure Numba to use RMM for memory allocations using the
Numba [EMM Plugin](https://numba.readthedocs.io/en/stable/cuda/external-memory.html#setting-emm-plugin).

This can be done in two ways:

1. Setting the environment variable `NUMBA_CUDA_MEMORY_MANAGER`:

  ```python
  $ NUMBA_CUDA_MEMORY_MANAGER=rmm.allocators.numba python (args)
  ```

2. Using the `set_memory_manager()` function provided by Numba:

  ```python
  >>> from numba import cuda
  >>> from rmm.allocators.numba import RMMNumbaManager
  >>> cuda.set_memory_manager(RMMNumbaManager)
  ```

**Note:** This only configures Numba to use the current RMM resource for allocations.
It does not initialize nor change the current resource, e.g., enabling a memory pool.
See [here](#memoryresource-objects) for more information on changing the current memory resource.

### Using RMM with PyTorch

[PyTorch](https://pytorch.org/docs/stable/notes/cuda.html) can use RMM
for memory allocation.  For example, to configure PyTorch to use an
RMM-managed pool:

```python
import rmm
from rmm.allocators.torch import rmm_torch_allocator
import torch

rmm.reinitialize(pool_allocator=True)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
```

PyTorch and RMM will now share the same memory pool.

You can, of course, use a custom memory resource with PyTorch as well:

```python
import rmm
from rmm.allocators.torch import rmm_torch_allocator
import torch

# note that you can configure PyTorch to use RMM either before or
# after changing RMM's memory resource.  PyTorch will use whatever
# memory resource is configured to be the "current" memory resource at
# the time of allocation.
torch.cuda.change_current_allocator(rmm_torch_allocator)

# configure RMM to use a managed memory resource, wrapped with a
# statistics resource adaptor that can report information about the
# amount of memory allocated:
mr = rmm.mr.StatisticsResourceAdaptor(rmm.mr.ManagedMemoryResource())
rmm.mr.set_current_device_resource(mr)

x = torch.tensor([1, 2]).cuda()

# the memory resource reports information about PyTorch allocations:
mr.allocation_counts
Out[6]:
{'current_bytes': 16,
 'current_count': 1,
 'peak_bytes': 16,
 'peak_count': 1,
 'total_bytes': 16,
 'total_count': 1}
```

## Taking ownership of C++ objects from Python

When interacting with a C++ library that uses RMM from Python, one
must be careful when taking ownership of `rmm::device_buffer` objects
on the Python side. The `rmm::device_buffer` does not contain an
owning reference to the memory resource used for its allocation (only
a `device_async_resource_ref`), and the allocating user is expected to
keep this memory resource alive for at least the lifetime of the
buffer. When taking ownership of such a buffer in Python, we have no
way (in the general case) of ensuring that the memory resource will
outlive the buffer we are now holding.

To avoid any issues, we need two things:

1. The C++ library we are interfacing with should accept a memory
   resource that is used for allocations that are returned to the
   user.
2. When calling into the library from python, we should provide a
   memory resource whose lifetime we control. This memory resource
   should then be provided when we take ownership of any allocated
   `rmm::device_buffer`s.

For example, suppose we have a C++ function that allocates
`device_buffer`s, which has a utility overload that defaults the
memory resource to the current device resource:

```c++
std::unique_ptr<rmm::device_buffer> allocate(
  std::size_t size,
  rmm::mr::device_async_resource_ref mr = get_current_device_resource())
{
    return std::make_unique<rmm::device_buffer>(size, rmm::cuda_stream_default, mr);
}
```

The Python `DeviceBuffer` class has a convenience Cython function,
`c_from_unique_ptr` to construct a `DeviceBuffer` from a
`unique_ptr<rmm::device_buffer>`, taking ownership of it. To do this
safely, we must ensure that the allocation that was done on the C++
side uses a memory resource we control. So:

```cython
# Bad, doesn't control lifetime
buffer_bad = DeviceBuffer.c_from_unique_ptr(allocate(10))

# Good, allocation happens with a memory resource we control
# mr is a DeviceMemoryResource
buffer_good = DeviceBuffer.c_from_unique_ptr(
    allocate(10, mr.get_mr()),
    mr=mr,
)
```

Note two differences between the bad and good cases:

1. In the good case we pass the memory resource to the allocation
   function.
2. In the good case, we pass _the same_ memory resource to the
   `DeviceBuffer` constructor so that its lifetime is tied to the
   lifetime of the buffer.

### Potential pitfalls of relying on `get_current_device_resource`

Functions in both the C++ and Python APIs that perform allocation
typically default the memory resource argument to the value of
`get_current_device_resource`. This is to simplify the interface for
callers. When using a C++ library from Python, this defaulting is
safe, _as long as_ it is only the Python process that ever calls
`set_current_device_resource`.

This is because the current device resource on the C++ side has a
lifetime which is expected to be managed by the user. The resources
set by `rmm::mr::set_current_device_resource` are stored in a static
`std::map` whose keys are device ids and values are raw pointers to
the memory resources. Consequently,
`rmm::mr::get_current_device_resource` returns an object with no
lifetime provenance. This is, for the reasons discussed above, not
usable from Python. To handle this on the Python side, the
Python-level `set_current_device_resource` sets the C++ resource _and_
stores the Python object in a static global dictionary. The Python
`get_current_device_resource` then _does not use_
`rmm::mr::get_current_device_resource` and instead looks up the
current device resource in this global dictionary.

Hence, if the C++ library we are interfacing with calls
`rmm::mr::set_current_device_resource`, the C++ and Python sides of
the program can disagree on what `get_current_device_resource`
returns. The only safe thing to do if using the simplified interfaces
is therefore to ensure that `set_current_device_resource` is only ever
called on the Python side.
