# RMM - the RAPIDS Memory Manager

Achieving optimal performance in GPU-centric workflows frequently requires
customizing how GPU ("device") memory is allocated.

RMM is a package that enables you to allocate device memory
in a highly configurable way. For example, it enables you to
allocate and use pools of GPU memory, or to use
[managed memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
for allocations.

You can also easily configure other libraries like Numba and CuPy
to use RMM for allocating device memory.

## Installation

See the project [README](https://github.com/rapidsai/rmm) for how to install RMM.

## Using RMM

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

Similarly, to use a pool of managed memory:

```python
>>> import rmm
>>> pool = rmm.mr.PoolMemoryResource(
...     upstream=rmm.mr.ManagedMemoryResource(),
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
