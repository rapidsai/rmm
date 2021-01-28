# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;RMM: RAPIDS Memory Manager</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/rmm/job/branches/job/rmm-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/rmm/job/branches/job/rmm-branch-pipeline/)


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
[below](#using-rmm-in-c++).

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/rmm/blob/main/README.md)
ensure you are on the `main` branch.

## Installation

### Conda

RMM can be installed with Conda ([miniconda](https://conda.io/miniconda.html), or the full
[Anaconda distribution](https://www.anaconda.com/download)) from the `rapidsai` channel:

```bash
# for CUDA 10.2
conda install -c nvidia -c rapidsai -c conda-forge -c defaults \
    rmm cudatoolkit=10.2
# for CUDA 10.1
conda install -c nvidia -c rapidsai -c conda-forge -c defaults \
    rmm cudatoolkit=10.1
# for CUDA 10.0
conda install -c nvidia -c rapidsai -c conda-forge -c defaults \
    rmm cudatoolkit=10.0
```

We also provide [nightly Conda packages](https://anaconda.org/rapidsai-nightly) built from the HEAD
of our latest development branch.

Note: RMM is supported only on Linux, and with Python versions 3.7 and later.

Note: The RMM package from Conda requires building with GCC 7 or later. Otherwise, your application may fail to build.

See the [Get RAPIDS version picker](https://rapids.ai/start.html) for more OS and version info.

## Building from Source

### Get RMM Dependencies

Compiler requirements:

* `gcc`     version 7.0 or higher required
* `nvcc`    version 9.0 or higher recommended
* `cmake`   version 3.18 or higher

CUDA/GPU requirements:

* CUDA 9.0+
* NVIDIA driver 396.44+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### Script to build RMM from source

To install RMM from source, ensure the dependencies are met and follow the steps below:

- Clone the repository and submodules
```bash
$ git clone --recurse-submodules https://github.com/rapidsai/rmm.git
$ cd rmm
```

Follow the instructions under "Create the conda development environment `cudf_dev`" in the
[cuDF README](https://github.com/rapidsai/cudf#build-from-source).

- Create the conda development environment `cudf_dev`
```bash
# create the conda environment (assuming in base `cudf` directory)
$ conda env create --name cudf_dev --file conda/environments/dev_py35.yml
# activate the environment
$ source activate cudf_dev
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
$ python setup.py build_ext --inplace
$ python setup.py install
$ pytest -v
```

Done! You are ready to develop for the RMM OSS project.

### Caching third-party dependencies

RMM uses [CPM.cmake](https://github.com/TheLartians/CPM.cmake) to
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
integrate RMM into your own CMake project. In your `CMakeLists.txt`, just add

```cmake
find_package(rmm [VERSION])
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC) rmm::rmm)
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
[Thrust's CMake documentation](https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md).

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

### Thread Safety

All current device memory resources are thread safe unless documented otherwise. More specifically,
calls to memory resource `allocate()` and `deallocate()` methods are safe with respect to calls to 
either of these functions from other threads. They are _not_ thread safe with respect to
construction and destruction of the memory resource object.

Note that a class `thread_safe_resource_adapter` is provided which can be used to adapt a memory
resource that is not thread safe to be thread safe (as described above). This adapter is not needed
with any current RMM device memory resources.

### Stream-ordered Memory Allocation

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

### Available Resources

RMM provides several `device_memory_resource` derived classes to satisfy various user requirements.
For more detailed information about these resources, see their respective documentation.

#### `cuda_memory_resource`

Allocates and frees device memory using `cudaMalloc` and `cudaFree`.

#### `managed_memory_resource`

Allocates and frees device memory using `cudaMallocManaged` and `cudaFree`. 

Note that `managed_memory_resource` cannot be used with NVIDIA Virtual GPU Software (vGPU, for use
with virtual machines or hypervisors) because [NVIDIA CUDA Unified Memory is not supported by 
NVIDIA vGPU](https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html#cuda-open-cl-support-vgpu).

#### `pool_memory_resource`

A coalescing, best-fit pool sub-allocator.

#### `fixed_size_memory_resource`

A memory resource that can only allocate a single fixed size. Average allocation and deallocation
cost is constant.

#### `binning_memory_resource`

Configurable to use multiple upstream memory resources for allocations that fall within different 
bin sizes. Often configured with multiple bins backed by `fixed_size_memory_resource`s and a single
`pool_memory_resource` for allocations larger than the largest bin size.

### Default Resources and Per-device Resources

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
   - Updates the default memory resource pointer for the current CUDA device to `new_resource`
   - Returns the previous default resource pointer
   - If `new_resource` is `nullptr`, then resets the default resource to `cuda_memory_resource`
   - This function is thread safe with respect to concurrent calls to it and
     `get_current_device_resource()`
   - For more explicit control, you can use `set_per_device_resource()`, which takes a device ID.

#### Example

```c++
rmm::mr::cuda_memory_resource cuda_mr;
// Construct a resource that uses a coalescing best-fit pool allocator
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`
rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(); // Points to `pool_mr`
```

#### Multiple Devices

A `device_memory_resource` should only be used when the active CUDA device is the same device
that was active when the `device_memory_resource` was created. Otherwise behavior is undefined.

If a `device_memory_resource` is used with a stream associated with a different CUDA device than the
device for which the memory resource was created, behavior is undefined.
 
Creating a `device_memory_resource` for each device requires care to set the current device before
creating each resource, and to maintain the lifetime of the resources as long as they are set as
per-device resources. Here is an example loop that creates `unique_ptr`s to `pool_memory_resource`
objects for each device and sets them as the per-device resource for that device.

```c++ 
std::vector<unique_ptr<pool_memory_resource>> per_device_pools;
for(int i = 0; i < N; ++i) {
    cudaSetDevice(i); // set device i before creating MR
    // Use a vector of unique_ptr to maintain the lifetime of the MRs
    per_device_pools.push_back(std::make_unique<pool_memory_resource>());
    // Set the per-device resource for device i
    set_per_device_resource(cuda_device_id{i}, &per_device_pools.back());
}
```

### Allocators

C++ interfaces commonly allow customizable memory allocation through an [`Allocator`](https://en.cppreference.com/w/cpp/named_req/Allocator) object. 
RMM provides several `Allocator` and `Allocator`-like classes.

#### `polymorphic_allocator`

A [stream-ordered](#stream-ordered-memory-allocation) allocator similar to [`std::pmr::polymorphic_allocator`](https://en.cppreference.com/w/cpp/memory/polymorphic_allocator). 
Unlike the standard C++ `Allocator` interface, the `allocate` and `deallocate` functions take a `cuda_stream_view` indicating the stream on which the (de)allocation occurs.

#### `stream_allocator_adaptor`

`stream_allocator_adaptor` can be used to adapt a stream-ordered allocator to present a standard `Allocator` interface to consumers that may not be designed to work with a stream-ordered interface.

Example:
```c++
rmm::cuda_stream stream;
rmm::mr::polymorphic_allocator<int> stream_alloc;

// Constructs an adaptor that forwards all (de)allocations to `stream_alloc` on `stream`.
auto adapted = rmm::mr::make_stream_allocator_adaptor(stream_alloc, stream);

// Allocates 100 bytes using `stream_alloc` on `stream`
auto p = adapted.allocate(100); 
...
// Deallocates using `stream_alloc` on `stream`
adapted.deallocate(p,100); 
```

#### `thrust_allocator`

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

1. `void* device_memory_resource::allocate(std::size_t bytes, std::size_t alignment)`
   - Returns a pointer to an allocation of at least `bytes` bytes aligned to the specified
     `alignment`

2. `void device_memory_resource::deallocate(void* p, std::size_t bytes, std::size_t alignment)`
   - Reclaims a previous allocation of size `bytes` pointed to by `p`. 


Unlike `device_memory_resource`, the `host_memory_resource` interface and behavior is identical to
`std::pmr::memory_resource`. 

### Available Resources

#### `new_delete_resource`

Uses the global `operator new` and `operator delete` to allocate host memory.

#### `pinned_memory_resource`

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

RMM includes a debug logger which can be enabled to log trace and debug information to a file. This 
information can show when errors occur, when additional memory is allocated from upstream resources,
etc. The default log file is `rmm_log.txt` in the current working directory, but the environment
variable `RMM_DEBUG_LOG_FILE` can be set to specify the path and file name.

There is a CMake configuration variable `RMM_LOGGING_LEVEL`, which can be set to enable compilation
of more detailed logging. The default is `INFO`. Available levels are `TRACE`, `DEBUG`, `INFO`,
`WARN`, `ERROR`, `CRITICAL` and `OFF`.

The log relies on the [spdlog](https://github.com/gabime/spdlog.git) library.

Note that to see logging below the `INFO` level, the C++ application must also call
`rmm::logger().set_level()`, e.g. to enable all levels of logging down to `TRACE`, call 
`rmm::logger().set_level(spdlog::level::trace)` (and compile with `-DRMM_LOGGING_LEVEL=TRACE`).

Note that debug logging is different from the CSV memory allocation logging provided by 
`rmm::mr::logging_resource_adapter`. The latter is for logging a history of allocation /
deallocation actions which can be useful for replay with RMM's replay benchmark.

## Using RMM in Python Code

There are two ways to use RMM in Python code:

1. Using the `rmm.DeviceBuffer` API to explicitly create and manage
   device memory allocations
2. Transparently via external libraries such as CuPy and Numba

RMM provides a `MemoryResource` abstraction to control _how_ device
memory is allocated in both the above uses.

### DeviceBuffers

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
>>> buf = rmm.to_device(a.tobytes())
>>> buf.size
24
```

Conversely, the data underlying a DeviceBuffer can be copied to the
host:

```python
>>> np.frombuffer(buf.tobytes())
array([1., 2., 3.])
```

### MemoryResource objects

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
...     upstream=rmm.mr.CudaMemoryResource(),
...     initial_pool_size=2**30,
...     maximum_pool_size=2**32
... )
>>> rmm.mr.set_current_device_resource(pool)
```
Other MemoryResources include:

* `FixedSizeMemoryResource` for allocating fixed blocks of memory
* `BinningMemoryResource` for allocating blocks within specified "bin" sizes from different memory 
resources

MemoryResources are highly configurable and can be composed together in different ways. 
See `help(rmm.mr)` for more information.

### Using RMM with CuPy

You can configure [CuPy](https://cupy.dev/) to use RMM for memory
allocations by setting the CuPy CUDA allocator to
`rmm_cupy_allocator`:

```python
>>> import rmm
>>> import cupy
>>> cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)
```

### Using RMM with Numba

You can configure Numba to use RMM for memory allocations using the
Numba [EMM Plugin](http://numba.pydata.org/numba-doc/latest/cuda/external-memory.html#setting-the-emm-plugin).

This can be done in two ways:

1. Setting the environment variable `NUMBA_CUDA_MEMORY_MANAGER`:

  ```python
  $ NUMBA_CUDA_MEMORY_MANAGER=rmm python (args)
  ```

2. Using the `set_memory_manager()` function provided by Numba:

  ```python
  >>> from numba import cuda
  >>> import rmm
  >>> cuda.set_memory_manager(rmm.RMMNumbaManager)
  ```
