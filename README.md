# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;RMM: RAPIDS Memory Manager</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/rmm/job/branches/job/rmm-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/rmm/job/branches/job/rmm-branch-pipeline/)


Achieving optimal performance in GPU-centric workflows frequently requires customizing how host and device memory are allocated. For example, using "pinned" host memory for asynchronous host <-> device memory transfers, or using a device memory pool sub-allocator to reduce the cost of dynamic device memory allocation. 

The goal of the RAPIDS Memory Manager (RMM) is to provide:
- A common interface that allows customizing [device](#device_memory_resource) and [host](#host_memory_resource) memory allocation
- A collection of [implementations](#available-resources) of the interface
- A collection of [data structures](#data-structures) that use the interface for memory allocation

For information on the interface RMM provides and how to use RMM in your C++ code, see [below](#using-rmm-in-c++).

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/rmm/blob/master/README.md) ensure you are on the `master` branch.

## Installation

### Conda

RMM can be installed with conda ([miniconda](https://conda.io/miniconda.html), or the full [Anaconda distribution](https://www.anaconda.com/download)) from the `rapidsai` channel:

For `rmm version == 0.14` :
```bash
# for CUDA 10.2
conda install -c nvidia -c rapidsai -c conda-forge -c defaults \
    rmm=0.14 cudatoolkit=10.2
```

For `rmm version == 0.12` :
```bash
# for CUDA 10.1
conda install -c nvidia -c rapidsai-nightly -c conda-forge -c defaults \
    rmm=0.12 python=3.6 cudatoolkit=10.1

# or, for CUDA 10.0
conda install -c nvidia -c rapidsai-nightly -c conda-forge -c defaults \
    rmm=0.12 python=3.6 cudatoolkit=10.0
```

For `rmm version == 0.11` :
```bash
# for CUDA 10.1
conda install -c nvidia -c rapidsai -c conda-forge -c defaults \
    rmm=0.11 python=3.6 cudatoolkit=10.1

# or, for CUDA 10.0
conda install -c nvidia -c rapidsai -c conda-forge -c defaults \
    rmm=0.11 python=3.6 cudatoolkit=10.0
```
We also provide [nightly conda packages](https://anaconda.org/rapidsai-nightly) built from the tip of our latest development branch.

Note: RMM is supported only on Linux, and with Python versions 3.6 or 3.7.

See the [Get RAPIDS version picker](https://rapids.ai/start.html) for more OS and version info.

## Building from Source

### Get RMM Dependencies

Compiler requirements:

* `gcc`     version 4.8 or higher recommended
* `nvcc`    version 9.0 or higher recommended
* `cmake`   version 3.12 or higher

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

Follow the instructions under "Create the conda development environment `cudf_dev`" in the [cuDF README](https://github.com/rapidsai/cudf#build-from-source).

- Create the conda development environment `cudf_dev`
```bash
# create the conda environment (assuming in base `cudf` directory)
$ conda env create --name cudf_dev --file conda/environments/dev_py35.yml
# activate the environment
$ source activate cudf_dev
```

- Build and install `librmm` using cmake & make. CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.
```bash

$ mkdir build                                       # make a build directory
$ cd build                                          # enter the build directory
$ cmake .. -DCMAKE_INSTALL_PREFIX=/install/path     # configure cmake ... use $CONDA_PREFIX if you're using Anaconda
$ make -j                                           # compile the library librmm.so ... '-j' will start a parallel job using the number of physical cores available on your system
$ make install                                      # install the library librmm.so to '/install/path'
```

- Building and installing `librmm` and `rmm` using build.sh. Build.sh creates build dir at root of git repository. build.sh depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.
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

# Using RMM in C++

The first goal of RMM is to provide a common interface for device and host memory allocation. 
This allows both _users_ and _implementers_ of custom allocation logic to program to a single interface.

To this end, RMM defines two abstract interface classes:
- [`rmm::mr::device_memory_resource`](#device_memory_resource) for device memory allocation
- [`rmm::mr::host_memory_resource`](#host_memory_resource) for host memory allocation

These classes are based on the [`std::pmr::memory_resource`](https://en.cppreference.com/w/cpp/memory/memory_resource) interface class introduced in C++17 for polymorphic memory allocation.

## `device_memory_resource`

`rmm::mr::device_memory_resource` is the base class that defines the interface for allocating and freeing device memory.

It has two key functions:

1. `void* device_memory_resource::allocate(std::size_t bytes, cudaStream_t s)`
   - Returns a pointer to an allocation of at least `bytes` bytes.

2. `void device_memory_resource::deallocate(void* p, std::size_t bytes, cudaStream_t s)`
   - Reclaims a previous allocation of size `bytes` pointed to by `p`. 
   - `p` *must* have been returned by a previous call to `allocate(bytes)`, otherwise behavior is undefined

It is up to a derived class to provide implementations of these functions. 
See [available resources](#availble-resources) for example `device_memory_resource` derived classes. 

Unlike `std::pmr::memory_resource`, `rmm::mr::device_memory_resource` does not allow specifying an alignment argument. 
All allocations are required to be aligned to at least 256B. 
Furthermore, `device_memory_resource` adds an additional `cudaStream_t` argument to allow specifying the stream on which to perform the (de)allocation.

### Available Resources

RMM provides several `device_memory_resource` derived classes to satisfy various user requirements.
For more detailed information about these resources, see their respective documentation.

#### `cuda_memory_resource`

Allocates and frees device memory using `cudaMalloc` and `cudaFree`.

#### `managed_memory_resource`

Allocates and frees device memory using `cudaMallocManaged` and `cudaFree`.

#### `cnmem_(managed_)memory_resource`

Uses the [CNMeM](https://github.com/NVIDIA/cnmem) pool sub-allocator to satisfy (de)allocations.

#### `pool_memory_resource`

A coalescing, best-fit pool sub-allocator.

### Default Resource

Frequently, users want to configure a `device_memory_resource` object once and use it for all allocations where another resource has not explicitly been provided. 
For example, one may want to construct a `pool_memory_resource` and use it for all allocations to get fast dynamic allocation.

To enable this use case, RMM provides the concept of a "default" `device_memory_resource`. 
This is the resource that will be used when another is not explicitly provided.

Accessing and modifying the default resource is done through two functions:
- `device_memory_resource* get_default_resource()`
   - Returns a pointer to the current default resource
   - The initial default memory resource is an instance of `cuda_memory_resource`
   - This function is thread safe

- `device_memory_resource* set_default_resource(device_memory_resource* new_resource)`
   - Updates the default memory resource pointer to `new_resource`
   - Returns the previous default resource pointer
   - If `new_resource` is `nullptr`, then returns the default resource to `cuda_memory_resource`
   - This function is thread safe

#### Example

```c++
rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(); // Points to `cuda_memory_resource`
rmm::mr::cnmem_memory_resource pool_mr{}; // Construct a resource that uses the CNMeM pool
rmm::mr::set_default_resource(&pool_mr); // Updates the default resource pointer to `pool_mr`
rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(); // Points to `pool_mr`
```

## Device Data Structures

### `device_buffer`

An untyped, unintialized RAII class for stream ordered device memory allocation.

#### Example

```c++
cudaStream_t s;
rmm::device_buffer b{100,s}; // Allocates at least 100 bytes on stream `s` using the *default* resource
void* p = b.data();          // Raw, untyped pointer to underlying device memory

kernel<<<..., s>>>(b.data()); // `b` is only safe to use on `s`

rmm::mr::device_memory_resource * mr = new my_custom_resource{...};
rmm::device_buffer b2{100, s, mr}; // Allocates at least 100 bytes on stream `s` using the explicitly provided resource
```

### `device_uvector<T>`
A typed, unintialized RAII class for allocation of a contiguous set of elements in device memory.
Similar to a `thrust::device_vector`, but as an optimization, does not default initialize the contained elements.
This optimization restricts the types `T` to trivially copyable types.

#### Example

```c++
cudaStream_t s;
rmm::device_uvector<int32_t> v(100, s); /// Allocates uninitialized storage for 100 `int32_t` elements on stream `s` using the default resource
thrust::uninitialized_fill(thrust::cuda::par.on(s), v.begin(), v.end(), int32_t{0}); // Initializes the elements to 0

rmm::mr::device_memory_resource * mr = new my_custom_resource{...};
rmm::device_vector<int32_t> v2{100, s, mr}; // Allocates uninitialized storage for 100 `int32_t` elements on stream `s` using the explicitly provided resource
```

### `device_scalar`
A typed, RAII class for allocation of a single element in device memory.
This is similar to a `device_uvector` with a single element, but provides convenience functions like modifying the value in device memory from the host, or retrieving the value from device to host.

#### Example
```c++
cudaStream_t s;
rmm::device_scalar<int32_t> a{s}; // Allocates uninitialized storage for a single `int32_t` in device memory
a.set_value(42, s); // Updates the value in device memory to `42` on stream `s`

kernel<<<...,s>>>(a.data()); // Pass raw pointer to underlying element in device memory

int32_t v = a.value(s); // Retrieves the value from device to host on stream `s`
```

## Using RMM with Thrust

RAPIDS and other CUDA libraries make heavy use of Thrust. Thrust uses CUDA device memory in two
situations:

 1. As the backing store for `thrust::device_vector`, and
 2. As temporary storage inside some algorithms, such as `thrust::sort`.

RMM provides `rmm::mr::thrust_allocator` as a conforming Thrust allocator that uses `device_memory_resource`s.

### Thrust Algorithms

To instruct a Thrust algorithm to use `rmm::mr::thrust_allocator` to allocate temporary storage, you can use the custom Thrust CUDA device execution policy: `rmm::exec_policy(stream)`.

`rmm::exec_policy(stream)` returns a `std::unique_ptr` to a Thrust execution policy that uses `rmm::mr::thrust_allocator` for temporary allocations.
In order to specify that the Thrust algorithm be executed on a specific stream, the usage is:

```c++
thrust::sort(rmm::exec_policy(stream)->on(stream), ...);
```

The first `stream` argument is the `stream` to use for `rmm::mr::thrust_allocator`.
The second `stream` argument is what should be used to execute the Thrust algorithm.
These two arguments must be identical.

## `host_memory_resource`

`rmm::mr::host_memory_resource` is the base class that defines the interface for allocating and freeing host memory.

Similar to `device_memory_resource`, it has two key functions for (de)allocation:

1. `void* device_memory_resource::allocate(std::size_t bytes, std::size_t alignment)`
   - Returns a pointer to an allocation of at least `bytes` bytes aligned to the specified `alignment`

2. `void device_memory_resource::deallocate(void* p, std::size_t bytes, std::size_t alignment)`
   - Reclaims a previous allocation of size `bytes` pointed to by `p`. 


Unlike `device_memory_resource`, the `host_memory_resource` interface and behavior is identical to `std::pmr::memory_resource`. 

### Available Resources

#### `new_delete_resource`

Uses the global `operator new` and `operator delete` to allocate host memory.

#### `pinned_memory_resource`

Allocates "pinned" host memory using `cuda(Malloc/Free)Host`.

## Host Data Structures

RMM does not currently provide any data structures that interface with `host_memory_resource`.
In the future, RMM will provide a similar host-side structure like `device_buffer` and an allocator that can be used with STL containers.



## Using RMM in Python Code

cuDF and other Python libraries typically create arrays of CUDA device memory
by using Numba's `cuda.device_array` interfaces. Until Numba provides a plugin
interface for using an external memory manager, RMM provides an API compatible
with `cuda.device_array` constructors that cuDF (also cuDF C++ API pytests)
should use to ensure all CUDA device memory is allocated via the memory manager.
RMM provides:

   - `rmm.device_array()`
   - `rmm.device_array_like()`
   - `rmm.to_device()`
   - `rmm.auto_device()`

Which are compatible with their Numba `cuda.*` equivalents. They return a Numba
NDArray object whose memory is allocated in CUDA device memory using RMM.

Following is an example from cuDF `groupby.py` that copies from a numpy array to
an equivalent CUDA `device_array` using `to_device()`, and creates a device
array using `device_array`, and then runs a Numba kernel (`group_mean`) to
compute the output values.

```
    ...
    dev_begins = rmm.to_device(np.asarray(begin))
    dev_out = rmm.device_array(size, dtype=np.float64)
    if size > 0:
        group_mean.forall(size)(sr.to_gpu_array(),
                                dev_begins,
                                dev_out)
    values[newk] = dev_out
```
In another example from cuDF `cudautils.py`, `fillna` uses `device_array_like`
to construct a CUDA device array with the same shape and data type as another.

```
def fillna(data, mask, value):
    out = rmm.device_array_like(data)
    out.copy_to_device(data)
    configured = gpu_fill_masked.forall(data.size)
    configured(value, mask, out)
    return out
```

`rmm` also provides `get_ipc_handle()` for getting the IPC handle associated
with a Numba NDArray, which accounts for the case where the data for the NDArray
is suballocated from some larger pool allocation by the memory manager.

### Handling RMM Options in Python Code

RMM currently defaults to just calling cudaMalloc, but you can enable the
experimental pool allocator by reinitializing RMM.

```
rmm.reinitialize(
    pool_allocator=False, # default is False
    managed_memory=False, # default is False
    initial_pool_size=int(2**31), # set to 2GiB. Default is 1/2 total GPU memory
    devices=0, # GPU device  IDs to register. By default registers only GPU 0.
    logging=True, # default is False -- has perf overhead
)
```

To configure RMM options to be used in cuDF before loading, simply do the above
before you `import cudf`. You can re-initialize the memory manager with
different settings at run time by calling `rmm.reinitialize()` with the above
options.

You can also optionally use the internal functions in cuDF which call these
functions. Here are some example configuration functions that can be used in
a notebook to initialize the memory manager in each Dask worker.

```python
import cudf


# Default passthrough to cudaMalloc
cudf.set_allocator()

# Use the pool allocator
cudf.set_allocator(pool=True)

# Use the pool allocator with a 2GiB initial pool size
cudf.set_allocator(pool=True, initial_pool_size=2<<30)
```

Remember that while the pool is in use memory is not freed. So if you follow
cuDF operations with device-memory-intensive computations that don't use RMM
(such as XGBoost), you will need to move the data to the host and then
finalize RMM. The Mortgage E2E workflow notebook uses this technique. We are
working on better ways to reclaim memory, as well as making RAPIDS machine
learning libraries use the same RMM memory pool.

### Memory info

The amount of free and total memory managed by RMM associated with a particular
stream can be obtained with the `get_info` function:

```python
meminfo = rmm.get_info()
print(meminfo.free)  # E.g. "16046292992"
print(meminfo.total) # E.g. "16914055168"
```
